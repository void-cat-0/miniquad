use crate::graphics::{ShaderError, ShaderMeta, ShaderType, UniformType};
use glsl_lang::{
    ast::*,
    parse::DefaultParse,
    transpiler::glsl,
    visitor::{HostMut, Visit, VisitorMut},
};
use glslang_sys as sys;
use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    ffi::{CStr, CString},
    ptr::NonNull,
    sync::OnceLock,
};

#[path = "shader_limits.rs"]
mod limits;

struct NativeShader {
    handle: NonNull<sys::glslang_shader_t>,
    // glslang keeps this source pointer until the shader is deleted.
    _source: CString,
}
impl Drop for NativeShader {
    fn drop(&mut self) {
        unsafe { sys::glslang_shader_delete(self.handle.as_ptr()) }
    }
}
struct NativeProgram(NonNull<sys::glslang_program_t>);
impl Drop for NativeProgram {
    fn drop(&mut self) {
        unsafe { sys::glslang_program_delete(self.0.as_ptr()) }
    }
}

fn stage(kind: ShaderType) -> sys::glslang_stage_t {
    match kind {
        ShaderType::Vertex => sys::glslang_stage_t::Vertex,
        ShaderType::Fragment => sys::glslang_stage_t::Fragment,
    }
}

fn error(kind: ShaderType, message: impl ToString) -> ShaderError {
    ShaderError::CompilationError {
        shader_type: kind,
        error_message: message.to_string(),
    }
}

fn input(code: &CString, kind: ShaderType, vulkan: bool) -> sys::glslang_input_t {
    sys::glslang_input_t {
        language: sys::glslang_source_t::GLSL,
        stage: stage(kind),

        client: if vulkan {
            sys::glslang_client_t::Vulkan
        } else {
            sys::glslang_client_t::None
        },
        client_version: sys::glslang_target_client_version_t::Vulkan1_1,
        target_language: if vulkan {
            sys::glslang_target_language_t::SPIRV
        } else {
            sys::glslang_target_language_t::None
        },

        target_language_version: sys::glslang_target_language_version_t::SPIRV1_3,
        code: code.as_ptr(),
        default_version: 100,
        default_profile: sys::glslang_profile_t::None,
        force_default_version_and_profile: 0,
        forward_compatible: 0,

        messages: if vulkan {
            sys::glslang_messages_t::SPV_RULES | sys::glslang_messages_t::VULKAN_RULES
        } else {
            sys::glslang_messages_t::DEFAULT
        },

        resource: &limits::DEFAULT_RESOURCES,
        callbacks: sys::glsl_include_callbacks_t {
            include_system: None,
            include_local: None,
            free_include_result: None,
        },
        callbacks_ctx: std::ptr::null_mut(),
    }
}

fn native(source: &str, kind: ShaderType, vulkan: bool) -> Result<NativeShader, ShaderError> {
    static INIT: OnceLock<bool> = OnceLock::new();
    if !*INIT.get_or_init(|| unsafe { sys::glslang_initialize_process() != 0 }) {
        return Err(error(kind, "Failed to initialize glslang"));
    }
    let code = CString::new(
        source
            .trim_start_matches('\u{feff}')
            .replace("\r\n", "\n")
            .replace('\r', "\n"),
    )?;
    let settings = input(&code, kind, vulkan);
    let handle = NonNull::new(unsafe { sys::glslang_shader_create(&settings) })
        .ok_or_else(|| error(kind, "Failed to allocate glslang shader"))?;
    let shader = NativeShader {
        handle,
        _source: code,
    };
    unsafe {
        sys::glslang_shader_set_options(
            shader.handle.as_ptr(),
            sys::glslang_shader_options_t::AUTO_MAP_LOCATIONS.0,
        );
        if sys::glslang_shader_preprocess(shader.handle.as_ptr(), &settings) == 0
            || (vulkan && sys::glslang_shader_parse(shader.handle.as_ptr(), &settings) == 0)
        {
            return Err(error(
                kind,
                CStr::from_ptr(sys::glslang_shader_get_info_log(shader.handle.as_ptr()))
                    .to_string_lossy(),
            ));
        }
    }
    Ok(shader)
}

fn parse(source: &str, kind: ShaderType) -> Result<TranslationUnit, ShaderError> {
    // glslang does not implement the WebGL draw-buffers extension. Its output
    // variables are translated below, so only remove that extension directive
    // before preprocessing, keeping line numbers and conditional compilation.
    let source = source
        .lines()
        .map(|line| {
            let directive = line.trim_start().strip_prefix('#').map(str::trim_start);
            if directive
                .and_then(|d| d.strip_prefix("extension"))
                .and_then(|d| d.trim_start().split(':').next())
                .is_some_and(|name| name.trim() == "GL_EXT_draw_buffers")
            {
                ""
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let shader = native(&source, kind, false)?;
    let expanded = unsafe {
        CStr::from_ptr(sys::glslang_shader_get_preprocessed_code(
            shader.handle.as_ptr(),
        ))
    }
    .to_string_lossy();
    TranslationUnit::parse(&expanded).map_err(|e| error(kind, e))
}

fn storage(qualifier: &TypeQualifier, value: StorageQualifierData) -> bool {
    qualifier.qualifiers.iter().any(|q| {
        matches!(&q.content,
        TypeQualifierSpecData::Storage(s) if s.content == value)
    })
}

fn layout_value(qualifier: &TypeQualifier, key: &str) -> Option<u32> {
    for q in &qualifier.qualifiers {
        let TypeQualifierSpecData::Layout(layout) = &q.content else {
            continue;
        };
        for id in &layout.ids {
            let LayoutQualifierSpecData::Identifier(name, Some(expr)) = &id.content else {
                continue;
            };
            if name.as_str() != key {
                continue;
            }
            match expr.content {
                ExprData::IntConst(value) => return Some(value as u32),
                ExprData::UIntConst(value) => return Some(value),
                _ => {}
            }
        }
    }
    None
}

fn set_layout(qualifier: &mut TypeQualifier, key: &str, value: u32) {
    for q in &mut qualifier.qualifiers {
        if let TypeQualifierSpecData::Layout(l) = &mut q.content {
            l.ids.retain(|i| !matches!(&i.content, LayoutQualifierSpecData::Identifier(n, _) if n.as_str() == key));
        }
    }
    qualifier.qualifiers.retain(|q| {
        !matches!(&q.content,
        TypeQualifierSpecData::Layout(layout) if layout.ids.is_empty())
    });
    qualifier.qualifiers.push(
        TypeQualifierSpecData::Layout(
            LayoutQualifierData {
                ids: vec![LayoutQualifierSpecData::Identifier(
                    IdentifierData::from(key).into(),
                    Some(Box::new(ExprData::UIntConst(value).into())),
                )
                .into()],
            }
            .into(),
        )
        .into(),
    );
}

fn split_declarations(unit: &mut TranslationUnit) {
    let mut items = Vec::new();
    for mut item in std::mem::take(&mut unit.0) {
        let mut extra = Vec::new();
        if let ExternalDeclarationData::Declaration(d) = &mut item.content {
            if let DeclarationData::InitDeclaratorList(list) = &mut d.content {
                for tail in std::mem::take(&mut list.tail) {
                    let tail = tail.content;
                    let ident = tail.ident.content;
                    let mut head = list.head.clone();
                    head.name = Some(ident.ident);
                    head.array_specifier = ident.array_spec;
                    head.initializer = tail.initializer;
                    extra.push(
                        ExternalDeclarationData::Declaration(
                            DeclarationData::InitDeclaratorList(
                                InitDeclaratorListData { head, tail: vec![] }.into(),
                            )
                            .into(),
                        )
                        .into(),
                    );
                }
            }
        }
        items.push(item);
        items.extend(extra);
    }
    unit.0 = items;
}

struct Upgrade {
    vertex: bool,
    color: bool,
    data: bool,
    instances: HashMap<String, Vec<String>>,
    sampler_aliases: HashMap<String, String>,
}
impl VisitorMut for Upgrade {
    fn visit_single_declaration(&mut self, declaration: &mut SingleDeclaration) -> Visit {
        if let Some(name) = &mut declaration.name {
            if let Some(alias) = self.sampler_aliases.get(name.as_str()) {
                *name = IdentifierData::from(alias.as_str()).into();
            }
        }
        Visit::Children
    }
    fn visit_single_declaration_no_type(
        &mut self,
        declaration: &mut SingleDeclarationNoType,
    ) -> Visit {
        let name = &mut declaration.ident.ident;
        if let Some(alias) = self.sampler_aliases.get(name.as_str()) {
            *name = IdentifierData::from(alias.as_str()).into();
        }
        Visit::Children
    }
    fn visit_function_parameter_declarator(
        &mut self,
        parameter: &mut FunctionParameterDeclarator,
    ) -> Visit {
        let name = &mut parameter.ident.ident;
        if let Some(alias) = self.sampler_aliases.get(name.as_str()) {
            *name = IdentifierData::from(alias.as_str()).into();
        }
        Visit::Children
    }
    fn visit_storage_qualifier(&mut self, q: &mut StorageQualifier) -> Visit {
        match q.content {
            StorageQualifierData::Attribute => q.content = StorageQualifierData::In,
            StorageQualifierData::Varying => {
                q.content = if self.vertex {
                    StorageQualifierData::Out
                } else {
                    StorageQualifierData::In
                }
            }
            _ => {}
        }
        Visit::Children
    }
    fn visit_expr(&mut self, expr: &mut Expr) -> Visit {
        if let ExprData::Dot(base, field) = &expr.content {
            if let ExprData::Variable(instance) = &base.content {
                if self
                    .instances
                    .get(instance.as_str())
                    .is_some_and(|members| members.iter().any(|m| m == field.as_str()))
                {
                    expr.content = ExprData::Variable(field.clone());
                }
            }
        }
        if let ExprData::Variable(id) = &mut expr.content {
            if let Some(alias) = self.sampler_aliases.get(id.as_str()) {
                *id = IdentifierData::from(alias.as_str()).into();
            }
            let replacement = match id.as_str() {
                "gl_FragColor" => {
                    self.color = true;
                    Some("_mq_FragColor")
                }
                "gl_FragData" => {
                    self.data = true;
                    Some("_mq_FragData")
                }
                "gl_FragDepthEXT" => Some("gl_FragDepth"),
                "gl_VertexID" => Some("gl_VertexIndex"),
                "gl_InstanceID" => Some("gl_InstanceIndex"),
                _ => None,
            };
            if let Some(name) = replacement {
                *id = IdentifierData::from(name).into();
            }
        }
        if let ExprData::FunCall(fun, args) = &mut expr.content {
            if let Some(id) = fun.as_ident_mut() {
                let replacement = match id.as_str() {
                    "texture1D" | "texture2D" | "texture3D" | "textureCube" => Some("texture"),
                    "texture1DProj" | "texture2DProj" | "texture3DProj" => Some("textureProj"),
                    "texture1DLod" | "texture2DLod" | "texture3DLod" | "textureCubeLod"
                    | "texture2DLodEXT" | "textureCubeLodEXT" => Some("textureLod"),
                    "texture2DProjLod" | "texture2DProjLodEXT" => Some("textureProjLod"),
                    "texture2DGradEXT" | "textureCubeGradEXT" | "texture2DGradARB" => {
                        Some("textureGrad")
                    }
                    "texture2DProjGradEXT" | "texture2DProjGradARB" => Some("textureProjGrad"),
                    _ => None,
                };
                if let Some(name) = replacement {
                    *id = IdentifierData::from(name).into();
                }
                if self.sampler_aliases.contains_key(id.as_str()) {
                    // Function identifiers are themselves Variable expressions.
                    // Visit arguments while keeping the upgraded built-in name
                    // separate from the sampler with that legacy identifier.
                    for arg in args {
                        arg.visit_mut(self);
                    }
                    return Visit::Parent;
                }
            }
        }
        Visit::Children
    }
}

fn uniform_type(value: UniformType) -> &'static str {
    match value {
        UniformType::Float1 => "float",
        UniformType::Float2 => "vec2",
        UniformType::Float3 => "vec3",
        UniformType::Float4 => "vec4",
        UniformType::Int1 => "int",
        UniformType::Int2 => "ivec2",
        UniformType::Int3 => "ivec3",
        UniformType::Int4 => "ivec4",
        UniformType::Mat4 => "mat4",
    }
}

fn upgrade(
    mut unit: TranslationUnit,
    kind: ShaderType,
    meta: &ShaderMeta,
    sampler_aliases: &HashMap<String, String>,
    uniform_arrays: &HashSet<String>,
) -> Result<TranslationUnit, ShaderError> {
    split_declarations(&mut unit);
    let mut visitor = Upgrade {
        vertex: matches!(kind, ShaderType::Vertex),
        color: false,
        data: false,
        instances: HashMap::new(),
        sampler_aliases: sampler_aliases.clone(),
    };
    let mut items = Vec::new();
    for mut item in unit.0 {
        match &mut item.content {
            ExternalDeclarationData::Preprocessor(p) => {
                if let PreprocessorData::Version(_) = p.content {
                    continue;
                }
                if let PreprocessorData::Extension(e) = &p.content {
                    let name = format!("{:?}", e.name);
                    if [
                        "GL_OES_standard_derivatives",
                        "GL_EXT_shader_texture_lod",
                        "GL_EXT_frag_depth",
                    ]
                    .iter()
                    .any(|n| name.contains(n))
                    {
                        continue;
                    }
                }
            }
            ExternalDeclarationData::Declaration(d) => match &mut d.content {
                DeclarationData::Precision(_, _) => continue,
                DeclarationData::Block(block)
                    if storage(&block.qualifier, StorageQualifierData::Uniform) =>
                {
                    if let Some(instance) = &block.identifier {
                        visitor.instances.insert(
                            instance.ident.as_str().to_owned(),
                            block
                                .fields
                                .iter()
                                .flat_map(|f| {
                                    f.identifiers.iter().map(|i| i.ident.as_str().to_owned())
                                })
                                .collect(),
                        );
                    }
                    continue;
                }
                DeclarationData::InitDeclaratorList(list) => {
                    let head = &mut list.head.content;
                    let ty = &mut head.ty.content;
                    if let Some(qualifier) = ty
                        .qualifier
                        .as_mut()
                        .filter(|q| storage(q, StorageQualifierData::Uniform))
                    {
                        let mut type_name = String::new();
                        glsl::show_type_specifier(&mut type_name, &ty.ty, &mut Default::default())
                            .unwrap();
                        if type_name.contains("sampler") {
                            let name = head
                                .name
                                .as_ref()
                                .ok_or_else(|| error(kind, "Sampler has no name"))?;
                            let slot = meta
                                .images
                                .iter()
                                .position(|n| n == name.as_str())
                                .ok_or_else(|| {
                                    error(
                                        kind,
                                        format!(
                                            "Texture {} is missing from ShaderMeta.images",
                                            name.as_str()
                                        ),
                                    )
                                })?;
                            set_layout(qualifier, "set", 0);
                            set_layout(
                                qualifier,
                                "binding",
                                super::IMAGE_BINDING_BASE + slot as u32,
                            );
                        } else {
                            continue;
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
        items.push(item);
    }
    unit.0 = items;
    unit.visit_mut(&mut visitor);
    if visitor.color && visitor.data {
        return Err(error(
            kind,
            "gl_FragColor and gl_FragData cannot share output location 0",
        ));
    }
    let mut globals = String::new();
    if !meta.uniforms.uniforms.is_empty() {
        globals.push_str("layout(std140, set=0, binding=0) uniform Uniforms {\n");
        for uniform in &meta.uniforms.uniforms {
            globals.push_str(&format!(
                "{} {}",
                uniform_type(uniform.uniform_type),
                uniform.name
            ));
            if uniform.array_count > 1 || uniform_arrays.contains(&uniform.name) {
                globals.push_str(&format!("[{}]", uniform.array_count));
            }
            globals.push_str(";\n");
        }
        globals.push_str("};\n");
    }
    if visitor.color {
        globals.push_str("layout(location=0) out vec4 _mq_FragColor;\n");
    }
    if visitor.data {
        globals.push_str("layout(location=0) out vec4 _mq_FragData[8];\n");
    }
    let generated = TranslationUnit::parse(&globals).map_err(|e| error(kind, e))?;
    let insert = unit
        .0
        .iter()
        .position(|d| !matches!(d.content, ExternalDeclarationData::Preprocessor(_)))
        .unwrap_or(unit.0.len());
    unit.0.splice(insert..insert, generated.0);
    Ok(unit)
}

fn sampler_aliases(
    vertex: &mut TranslationUnit,
    fragment: &mut TranslationUnit,
    meta: &ShaderMeta,
) -> HashMap<String, String> {
    struct Identifiers(HashSet<String>);
    impl VisitorMut for Identifiers {
        fn visit_identifier(&mut self, name: &mut Identifier) -> Visit {
            self.0.insert(name.as_str().to_owned());
            Visit::Children
        }
    }
    let mut identifiers = Identifiers(HashSet::new());
    vertex.visit_mut(&mut identifiers);
    fragment.visit_mut(&mut identifiers);
    identifiers.0.extend(meta.images.iter().cloned());
    identifiers.0.extend(
        meta.uniforms
            .uniforms
            .iter()
            .map(|uniform| uniform.name.clone()),
    );
    let mut aliases = HashMap::new();
    for (slot, name) in meta.images.iter().enumerate() {
        // ESSL permits a sampler named `texture`. After upgrading texture2D
        // to texture(), that variable would hide the modern built-in function.
        // Keep the public image name and descriptor slot; only GLSL identifiers
        // change. Choose one collision-free alias shared by both shader stages.
        if ![
            "texture",
            "textureProj",
            "textureLod",
            "textureProjLod",
            "textureGrad",
            "textureProjGrad",
        ]
        .contains(&name.as_str())
        {
            continue;
        }
        let base = format!("_mq_sampler_{}", super::IMAGE_BINDING_BASE + slot as u32);
        let mut alias = base.clone();
        let mut suffix = 0;
        while !identifiers.0.insert(alias.clone()) {
            suffix += 1;
            alias = format!("{base}_{suffix}");
        }
        aliases.insert(name.clone(), alias);
    }
    aliases
}

fn constant_integer(expr: &Expr, constants: &HashMap<String, i64>) -> Option<i64> {
    match &expr.content {
        ExprData::IntConst(value) => Some(*value as i64),
        ExprData::UIntConst(value) => Some(*value as i64),
        ExprData::BoolConst(value) => Some(i64::from(*value)),
        ExprData::Variable(name) => constants.get(name.as_str()).copied(),
        ExprData::Unary(op, expr) => {
            let value = constant_integer(expr, constants)?;
            match op.content {
                UnaryOpData::Add => Some(value),
                UnaryOpData::Minus => value.checked_neg(),
                UnaryOpData::Complement => Some(!value),
                UnaryOpData::Not => Some(i64::from(value == 0)),
                _ => None,
            }
        }
        ExprData::Binary(op, left, right) => {
            let left = constant_integer(left, constants)?;
            let right = constant_integer(right, constants)?;
            use BinaryOpData::*;
            match op.content {
                Add => left.checked_add(right),
                Sub => left.checked_sub(right),
                Mult => left.checked_mul(right),
                Div => left.checked_div(right),
                Mod => left.checked_rem(right),
                LShift => u32::try_from(right)
                    .ok()
                    .and_then(|shift| left.checked_shl(shift)),
                RShift => u32::try_from(right)
                    .ok()
                    .and_then(|shift| left.checked_shr(shift)),
                BitOr => Some(left | right),
                BitXor => Some(left ^ right),
                BitAnd => Some(left & right),
                Equal => Some(i64::from(left == right)),
                NonEqual => Some(i64::from(left != right)),
                Lt => Some(i64::from(left < right)),
                Gt => Some(i64::from(left > right)),
                Lte => Some(i64::from(left <= right)),
                Gte => Some(i64::from(left >= right)),
                And => Some(i64::from(left != 0 && right != 0)),
                Or => Some(i64::from(left != 0 || right != 0)),
                Xor => Some(i64::from((left != 0) != (right != 0))),
            }
        }
        ExprData::Ternary(condition, yes, no) => constant_integer(
            if constant_integer(condition, constants)? != 0 {
                yes
            } else {
                no
            },
            constants,
        ),
        _ => None,
    }
}

fn global_constants(unit: &TranslationUnit) -> HashMap<String, i64> {
    let mut constants = HashMap::new();
    for item in &unit.0 {
        let ExternalDeclarationData::Declaration(declaration) = &item.content else {
            continue;
        };
        let DeclarationData::InitDeclaratorList(list) = &declaration.content else {
            continue;
        };
        let head = &list.head;
        let (Some(qualifier), Some(name), Some(initializer)) =
            (&head.ty.qualifier, &head.name, &head.initializer)
        else {
            continue;
        };
        if !storage(qualifier, StorageQualifierData::Const) {
            continue;
        }
        if let InitializerData::Simple(expr) = &initializer.content {
            if let Some(value) = constant_integer(expr, &constants) {
                constants.insert(name.as_str().to_owned(), value);
            }
        }
    }
    constants
}

fn location_width(
    head: &SingleDeclaration,
    constants: &HashMap<String, i64>,
    kind: ShaderType,
) -> Result<u32, ShaderError> {
    use TypeSpecifierNonArrayData::*;
    let columns = match head.ty.ty.ty.content {
        Mat2 | Mat22 | Mat23 | Mat24 | DMat2 | DMat22 | DMat23 | DMat24 => 2,
        Mat3 | Mat32 | Mat33 | Mat34 | DMat3 | DMat32 | DMat33 | DMat34 => 3,
        Mat4 | Mat42 | Mat43 | Mat44 | DMat4 | DMat42 | DMat43 | DMat44 => 4,
        _ => 1,
    };
    let mut width: u32 = columns;
    for array in [&head.array_specifier, &head.ty.ty.array_specifier]
        .iter()
        .filter_map(|array| array.as_ref())
    {
        for dimension in &array.dimensions {
            if let ArraySpecifierDimensionData::ExplicitlySized(expr) = &dimension.content {
                let count = constant_integer(expr, constants)
                    .and_then(|value| u32::try_from(value).ok())
                    .filter(|&value| value > 0)
                    .ok_or_else(|| error(kind, "Cannot determine shader interface array size"))?;
                width = width
                    .checked_mul(count)
                    .ok_or_else(|| error(kind, "Shader interface array size overflows u32"))?;
            }
        }
    }
    Ok(width)
}

fn interface_locations(
    vertex: &mut TranslationUnit,
    fragment: &mut TranslationUnit,
) -> Result<(), ShaderError> {
    // Match names across stages, including when declarations occur in different orders.
    let mut locations = HashMap::new();
    let mut next = 0;
    for (unit, direction, kind) in [
        (&*vertex, StorageQualifierData::Out, ShaderType::Vertex),
        (&*fragment, StorageQualifierData::In, ShaderType::Fragment),
    ] {
        let constants = global_constants(unit);
        for item in &unit.0 {
            let ExternalDeclarationData::Declaration(d) = &item.content else {
                continue;
            };
            let DeclarationData::InitDeclaratorList(list) = &d.content else {
                continue;
            };
            let (Some(q), Some(name)) = (&list.head.ty.qualifier, &list.head.name) else {
                continue;
            };
            if !storage(q, direction.clone()) {
                continue;
            }
            let Some(location) = layout_value(q, "location") else {
                continue;
            };
            locations.insert(name.as_str().to_owned(), location);
            next = next.max(
                location
                    .checked_add(location_width(&list.head, &constants, kind)?)
                    .ok_or_else(|| error(kind, "Shader interface locations overflow u32"))?,
            );
        }
    }
    for (unit, direction, kind) in [
        (vertex, StorageQualifierData::Out, ShaderType::Vertex),
        (fragment, StorageQualifierData::In, ShaderType::Fragment),
    ] {
        let constants = global_constants(unit);
        for item in &mut unit.0 {
            let ExternalDeclarationData::Declaration(d) = &mut item.content else {
                continue;
            };
            let DeclarationData::InitDeclaratorList(list) = &mut d.content else {
                continue;
            };
            let width = if list
                .head
                .ty
                .qualifier
                .as_ref()
                .is_some_and(|q| storage(q, direction.clone()))
            {
                location_width(&list.head, &constants, kind)?
            } else {
                continue;
            };
            let head = &mut list.head.content;
            let (Some(q), Some(name)) = (&mut head.ty.qualifier, &head.name) else {
                continue;
            };
            if !storage(q, direction.clone()) || layout_value(q, "location").is_some() {
                continue;
            }
            let location = if let Some(&location) = locations.get(name.as_str()) {
                location
            } else {
                let location = next;
                next = next
                    .checked_add(width)
                    .ok_or_else(|| error(kind, "Shader interface locations overflow u32"))?;
                locations.insert(name.as_str().to_owned(), location);
                location
            };
            set_layout(q, "location", location);
        }
    }
    Ok(())
}

fn remap_vertex_depth(unit: &mut TranslationUnit) -> Result<(), ShaderError> {
    struct Identifiers(HashSet<String>);
    impl VisitorMut for Identifiers {
        fn visit_identifier(&mut self, name: &mut Identifier) -> Visit {
            self.0.insert(name.as_str().to_owned());
            Visit::Children
        }
    }
    let mut identifiers = Identifiers(HashSet::new());
    unit.visit_mut(&mut identifiers);
    let mut entry = "_mq_vertex_main".to_owned();
    let mut suffix = 0;
    while identifiers.0.contains(&entry) {
        suffix += 1;
        entry = format!("_mq_vertex_main_{suffix}");
    }
    let mut definitions = 0;
    for item in &mut unit.0 {
        let prototype = match &mut item.content {
            ExternalDeclarationData::FunctionDefinition(function) => {
                if function.prototype.name.as_str() == "main" {
                    definitions += 1;
                }
                Some(&mut function.prototype)
            }
            ExternalDeclarationData::Declaration(declaration) => match &mut declaration.content {
                DeclarationData::FunctionPrototype(prototype) => Some(prototype),
                _ => None,
            },
            _ => None,
        };
        if let Some(prototype) = prototype.filter(|prototype| prototype.name.as_str() == "main") {
            prototype.name = IdentifierData::from(entry.as_str()).into();
        }
    }
    if definitions != 1 {
        return Err(error(
            ShaderType::Vertex,
            "Vertex shader must define exactly one main function",
        ));
    }
    // All Shader::new GLSL versions use OpenGL's -w..w clip depth. Vulkan
    // clips to 0..w, so translate only the final GPU position. A wrapper also
    // covers early returns while preserving the source's varying calculations.
    let wrapper = TranslationUnit::parse(&format!(
        "void main() {{ {entry}(); gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5; }}"
    ))
    .map_err(|error_value| error(ShaderType::Vertex, error_value))?;
    unit.0.extend(wrapper.0);
    Ok(())
}

/// Translate a vertex/fragment pair from GLSL 1.10–4.60 or ESSL 1.00–3.20.
/// The rendering API still defines the available resources through ShaderMeta.
pub fn translate_glsl(
    vertex: &str,
    fragment: &str,
    meta: &ShaderMeta,
) -> Result<(String, String), ShaderError> {
    let (vertex, fragment, _) = translate_with_layout(vertex, fragment, meta)?;
    Ok((vertex, fragment))
}

fn uniform_arrays(unit: &TranslationUnit, names: &mut HashSet<String>) {
    for item in &unit.0 {
        let ExternalDeclarationData::Declaration(declaration) = &item.content else {
            continue;
        };
        match &declaration.content {
            DeclarationData::InitDeclaratorList(list) => {
                let head = &list.head;
                if head
                    .ty
                    .qualifier
                    .as_ref()
                    .is_some_and(|q| storage(q, StorageQualifierData::Uniform))
                    && (head.array_specifier.is_some() || head.ty.ty.array_specifier.is_some())
                {
                    if let Some(name) = &head.name {
                        names.insert(name.as_str().to_owned());
                    }
                }
            }
            DeclarationData::Block(block)
                if storage(&block.qualifier, StorageQualifierData::Uniform) =>
            {
                for field in &block.fields {
                    for identifier in &field.identifiers {
                        if field.ty.array_specifier.is_some() || identifier.array_spec.is_some() {
                            names.insert(identifier.ident.as_str().to_owned());
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn translate_with_layout(
    vertex: &str,
    fragment: &str,
    meta: &ShaderMeta,
) -> Result<(String, String, HashSet<String>), ShaderError> {
    let mut vertex = parse(vertex, ShaderType::Vertex)?;
    let mut fragment = parse(fragment, ShaderType::Fragment)?;
    split_declarations(&mut vertex);
    split_declarations(&mut fragment);
    // Metadata gives the element count but does not distinguish a scalar from
    // an array of one element. Preserve that distinction from either stage.
    let mut arrays = HashSet::new();
    uniform_arrays(&vertex, &mut arrays);
    uniform_arrays(&fragment, &mut arrays);
    let aliases = sampler_aliases(&mut vertex, &mut fragment, meta);
    let mut v = upgrade(vertex, ShaderType::Vertex, meta, &aliases, &arrays)?;
    let mut f = upgrade(fragment, ShaderType::Fragment, meta, &aliases, &arrays)?;
    interface_locations(&mut v, &mut f)?;
    remap_vertex_depth(&mut v)?;
    let emit = |unit: &TranslationUnit| {
        let mut source = String::from("#version 460 core\n");
        glsl::show_translation_unit(&mut source, unit, Default::default()).unwrap();
        source
    };
    Ok((emit(&v), emit(&f), arrays))
}

/// Compile a GLSL/ESSL pair to SPIR-V without allocating Vulkan resources.
pub fn compile_glsl(
    vertex: &str,
    fragment: &str,
    meta: &ShaderMeta,
) -> Result<(Vec<u32>, Vec<u32>), ShaderError> {
    let (vertex, fragment, _) = compile_glsl_with_layout(vertex, fragment, meta)?;
    Ok((vertex, fragment))
}

pub(super) fn compile_glsl_with_layout(
    vertex: &str,
    fragment: &str,
    meta: &ShaderMeta,
) -> Result<(Vec<u32>, Vec<u32>, HashSet<String>), ShaderError> {
    let (vertex, fragment, arrays) = translate_with_layout(vertex, fragment, meta)?;
    let v = native(&vertex, ShaderType::Vertex, true)?;
    let f = native(&fragment, ShaderType::Fragment, true)?;
    let program = NativeProgram(
        NonNull::new(unsafe { sys::glslang_program_create() })
            .ok_or_else(|| ShaderError::LinkError("Failed to allocate glslang program".into()))?,
    );
    unsafe {
        sys::glslang_program_add_shader(program.0.as_ptr(), v.handle.as_ptr());
        sys::glslang_program_add_shader(program.0.as_ptr(), f.handle.as_ptr());
        if sys::glslang_program_link(
            program.0.as_ptr(),
            (sys::glslang_messages_t::SPV_RULES | sys::glslang_messages_t::VULKAN_RULES).0,
        ) == 0
            || sys::glslang_program_map_io(program.0.as_ptr()) == 0
        {
            return Err(ShaderError::LinkError(
                CStr::from_ptr(sys::glslang_program_get_info_log(program.0.as_ptr()))
                    .to_string_lossy()
                    .into_owned(),
            ));
        }
        let compile = |kind| -> Result<Vec<u32>, ShaderError> {
            let mut options = sys::glslang_spv_options_t {
                generate_debug_info: false,
                strip_debug_info: true,
                disable_optimizer: true,
                optimize_size: false,
                disassemble: false,
                validate: false,
                emit_nonsemantic_shader_debug_info: false,
                emit_nonsemantic_shader_debug_source: false,
                compile_only: false,
                optimize_allow_expanded_id_bound: false,
            };
            sys::glslang_program_SPIRV_generate_with_options(
                program.0.as_ptr(),
                stage(kind),
                &mut options,
            );
            let size = sys::glslang_program_SPIRV_get_size(program.0.as_ptr());
            if size == 0 {
                return Err(error(
                    kind,
                    CStr::from_ptr(sys::glslang_program_SPIRV_get_messages(program.0.as_ptr()))
                        .to_string_lossy(),
                ));
            }
            let mut words = vec![0; size];
            sys::glslang_program_SPIRV_get(program.0.as_ptr(), words.as_mut_ptr());
            Ok(words)
        };
        Ok((
            compile(ShaderType::Vertex)?,
            compile(ShaderType::Fragment)?,
            arrays,
        ))
    }
}
