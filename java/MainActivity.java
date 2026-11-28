package TARGET_PACKAGE_NAME;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.app.Activity;
import android.os.Bundle;
import android.os.Build;
import android.util.Log;
import android.util.TypedValue;

import android.view.View;
import android.view.ViewGroup;
import android.view.Surface;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowManager.LayoutParams;
import android.view.SurfaceView;
import android.view.SurfaceHolder;
import android.view.MotionEvent;
import android.view.KeyEvent;
import android.view.ActionMode;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowInsetsController;
import android.view.inputmethod.InputMethodManager;

import android.content.Context;
import android.content.Intent;
import android.content.res.Configuration;
import android.content.ClipData;
import android.content.ClipboardManager;

import android.graphics.Color;
import android.graphics.Insets;
import android.view.inputmethod.InputConnection;
import android.view.inputmethod.EditorInfo;
import android.widget.LinearLayout;
import android.widget.FrameLayout;
import android.widget.EditText;
import android.widget.TextView;
import android.text.Editable;
import android.text.Selection;
import android.text.TextWatcher;
import android.text.InputType;

import quad_native.QuadNative;

// note: //% is a special miniquad's pre-processor for plugins
// when there are no plugins - //% whatever will be replaced to an empty string
// before compiling

//% IMPORTS

class HiddenEditText extends EditText {
    private boolean mIgnoreUpdates = false;
    private boolean mInitialized = false;

    public HiddenEditText(Context context) {
        super(context);
        setBackgroundColor(Color.TRANSPARENT);
        setTextColor(Color.TRANSPARENT);
        setAlpha(0.01f); // Visible to WindowManager, but transparent to user
        setFocusable(true);
        setFocusableInTouchMode(true);
        setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_YES);
        setLayoutParams(new android.widget.FrameLayout.LayoutParams(1, 1));

        addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {}

            @Override
            public void afterTextChanged(Editable s) {
                sendStateToRust();
            }
        });

        setOnEditorActionListener(new TextView.OnEditorActionListener() {
            @Override
            public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
                if (MainActivity.sTextInputIsMultiline && actionId == EditorInfo.IME_NULL) {
                    // Enter in a multiline field inserts a newline.
                    return false;
                }
                QuadNative.surfaceOnImeAction(actionId, MainActivity.sFocusedElementId);
                return true;
            }
        });

        mInitialized = true;
    }

    @Override
    public boolean dispatchKeyEvent(KeyEvent event) {
        return super.dispatchKeyEvent(event);
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_DPAD_UP || keyCode == KeyEvent.KEYCODE_DPAD_DOWN
                || keyCode == KeyEvent.KEYCODE_DPAD_LEFT || keyCode == KeyEvent.KEYCODE_DPAD_RIGHT
                || keyCode == KeyEvent.KEYCODE_MOVE_HOME || keyCode == KeyEvent.KEYCODE_MOVE_END
                || keyCode == KeyEvent.KEYCODE_PAGE_UP || keyCode == KeyEvent.KEYCODE_PAGE_DOWN) {
            if (event.isShiftPressed()) {
                QuadNative.surfaceOnKeyDown(KeyEvent.KEYCODE_SHIFT_LEFT);
            }
            QuadNative.surfaceOnKeyDown(keyCode);
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }

    @Override
    public boolean onKeyUp(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_DPAD_UP || keyCode == KeyEvent.KEYCODE_DPAD_DOWN
                || keyCode == KeyEvent.KEYCODE_DPAD_LEFT || keyCode == KeyEvent.KEYCODE_DPAD_RIGHT
                || keyCode == KeyEvent.KEYCODE_MOVE_HOME || keyCode == KeyEvent.KEYCODE_MOVE_END
                || keyCode == KeyEvent.KEYCODE_PAGE_UP || keyCode == KeyEvent.KEYCODE_PAGE_DOWN) {
            QuadNative.surfaceOnKeyUp(keyCode);
            if (!event.isShiftPressed()) {
                QuadNative.surfaceOnKeyUp(KeyEvent.KEYCODE_SHIFT_LEFT);
            }
            return true;
        }
        return super.onKeyUp(keyCode, event);
    }

    @Override
    protected void onSelectionChanged(int selStart, int selEnd) {
        super.onSelectionChanged(selStart, selEnd);
        sendStateToRust();
    }

    public void setIgnoreUpdates(boolean ignore) {
        mIgnoreUpdates = ignore;
    }

    public void applyStateFromRust(String text, int selectionStart, int selectionEnd, int composingStart, int composingEnd) {
        Editable editable = getText();
        if (editable == null) return;
        mIgnoreUpdates = true;
        try {
            int currentComposingStart = android.view.inputmethod.BaseInputConnection.getComposingSpanStart(editable);
            int currentComposingEnd = android.view.inputmethod.BaseInputConnection.getComposingSpanEnd(editable);
            boolean textChanged = !editable.toString().equals(text);
            boolean composingEnded = composingStart < 0 && currentComposingStart >= 0;
            if (textChanged || composingEnded) {
                android.view.inputmethod.BaseInputConnection.removeComposingSpans(editable);
            }
            if (textChanged) {
                editable.replace(0, editable.length(), text);
            }
            int len = editable.length();
            int start = Math.max(0, Math.min(len, selectionStart));
            int end = Math.max(0, Math.min(len, selectionEnd));
            Selection.setSelection(editable, start, end);
            InputMethodManager imm = (InputMethodManager) getContext().getSystemService(Context.INPUT_METHOD_SERVICE);
            if (imm != null) {
                if (textChanged || composingEnded) {
                    imm.restartInput(this);
                } else {
                    imm.updateSelection(this, Math.min(start, end), Math.max(start, end),
                        currentComposingStart, currentComposingEnd);
                }
            }
        } finally {
            mIgnoreUpdates = false;
        }
    }

    private void sendStateToRust() {
        if (!mInitialized || mIgnoreUpdates) return;
        // No session, or a new session whose text has not arrived yet: the
        // editable still holds an earlier field's text.
        if (MainActivity.sFocusedElementId <= 0 || MainActivity.sAwaitingState) return;

        Editable editable = getText();
        if (editable == null) return;

        String text = editable.toString();
        int selectionStart = Math.max(0, Selection.getSelectionStart(editable));
        int selectionEnd = Math.max(0, Selection.getSelectionEnd(editable));
        int composingStart = android.view.inputmethod.BaseInputConnection.getComposingSpanStart(editable);
        int composingEnd = android.view.inputmethod.BaseInputConnection.getComposingSpanEnd(editable);

        MainActivity.sTextInputText = text;
        MainActivity.sTextInputSelectionStart = selectionStart;
        MainActivity.sTextInputSelectionEnd = selectionEnd;

        QuadNative.surfaceOnImeStateChanged(text, selectionStart, selectionEnd, composingStart, composingEnd, MainActivity.sFocusedElementId);
    }
}

class QuadSurface
    extends
        SurfaceView
    implements
        View.OnTouchListener,
        View.OnKeyListener,
        SurfaceHolder.Callback {

    public QuadSurface(Context context){
        super(context);
        getHolder().addCallback(this);

        setFocusable(true);
        setFocusableInTouchMode(true);
        requestFocus();
        setOnTouchListener(this);
        setOnKeyListener(this);
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        QuadNative.surfaceOnSurfaceCreated(getNativeSurface());
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        QuadNative.surfaceOnSurfaceDestroyed(getNativeSurface());
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        QuadNative.surfaceOnSurfaceChanged(getNativeSurface(), width, height);
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        int pointerCount = event.getPointerCount();

        int action = event.getActionMasked();
        switch (action) {
        case MotionEvent.ACTION_DOWN:
        case MotionEvent.ACTION_POINTER_DOWN: {
            int pointerIndex = event.getActionIndex();
            int id = event.getPointerId(pointerIndex);
            final float x = event.getX(pointerIndex);
            final float y = event.getY(pointerIndex);
            // Like native fields, a new touch dismisses the text toolbar;
            // the application shows it again after a long press.
            if (MainActivity.sInstance != null) {
                MainActivity.sInstance.finishTextMenu();
            }
            QuadNative.surfaceOnTouch(id, 0, x, y, event.getEventTime());

            break;
        }
        case MotionEvent.ACTION_MOVE: {
            for (int i = 0; i < pointerCount; i++) {
                final int id = event.getPointerId(i);
                final float x = event.getX(i);
                final float y = event.getY(i);
                QuadNative.surfaceOnTouch(id, 1, x, y, event.getEventTime());
            }
            break;
        }
        case MotionEvent.ACTION_UP:
        case MotionEvent.ACTION_POINTER_UP: {
            int pointerIndex = event.getActionIndex();
            int id = event.getPointerId(pointerIndex);
            final float x = event.getX(pointerIndex);
            final float y = event.getY(pointerIndex);
            if (MainActivity.sHiddenEditText != null && MainActivity.sFocusedElementId > 0) {
                InputMethodManager imm = (InputMethodManager) getContext().getSystemService(Context.INPUT_METHOD_SERVICE);
                if (imm != null) {
                    imm.viewClicked(MainActivity.sHiddenEditText);
                }
            }
            QuadNative.surfaceOnTouch(id, 2, x, y, event.getEventTime());
            break;
        }
        case MotionEvent.ACTION_CANCEL: {
            for (int i = 0; i < pointerCount; i++) {
                final int id = event.getPointerId(i);
                final float x = event.getX(i);
                final float y = event.getY(i);
                QuadNative.surfaceOnTouch(id, 3, x, y, event.getEventTime());
            }
            break;
        }
        default:
            break;
        }

        return true;
    }

    @SuppressWarnings("deprecation")
    @Override
    public boolean onKey(View v, int keyCode, KeyEvent event) {
        // Back goes through Activity.onBackPressed, which forwards it.
        if (keyCode == KeyEvent.KEYCODE_BACK) {
            return false;
        }
        if (MainActivity.sFocusedElementId > 0) {
            return false;
        }

        if (event.getAction() == KeyEvent.ACTION_DOWN && keyCode != 0) {
            QuadNative.surfaceOnKeyDown(keyCode);
        }

        if (event.getAction() == KeyEvent.ACTION_UP && keyCode != 0) {
            QuadNative.surfaceOnKeyUp(keyCode);
        }

        if (event.getAction() == KeyEvent.ACTION_UP || event.getAction() == KeyEvent.ACTION_MULTIPLE) {
            int character = event.getUnicodeChar();
            if (character == 0) {
                String characters = event.getCharacters();
                if (characters != null && !characters.isEmpty()) {
                    character = characters.charAt(0);
                }
            }

            if (character != 0) {
                QuadNative.surfaceOnCharacter(character);
            }
        }

        return true;
    }

    @Override
    public InputConnection onCreateInputConnection(EditorInfo outAttrs) {
        if (MainActivity.sHiddenEditText != null && MainActivity.sFocusedElementId > 0) {
            return MainActivity.sHiddenEditText.onCreateInputConnection(outAttrs);
        }
        return null;
    }

    public Surface getNativeSurface() {
        return getHolder().getSurface();
    }
}

class ResizingLayout
    extends
        FrameLayout
    implements
        View.OnApplyWindowInsetsListener {
    //% RESIZING_LAYOUT_BODY

    public ResizingLayout(MainActivity activity){
        super(activity);
        setBackgroundColor(Color.BLACK);
        setOnApplyWindowInsetsListener(this);

        //% RESIZING_LAYOUT_CONSTRUCTOR
    }

    private int mKeyboardHeight = -1;

    // The surface keeps its size while the soft keyboard is open: only the
    // system bars pad it. The keyboard is reported as a rectangle covering
    // the bottom of the surface, and the application scrolls the focused
    // field into view. Resizing the surface instead would change the
    // application's layout (and anything that scales with the window) every
    // time the keyboard opens, which is most visible in landscape.
    @Override
    @SuppressWarnings("deprecation")
    public WindowInsets onApplyWindowInsets(View v, WindowInsets insets) {
        int left, top, right, bottom, ime;
        if (Build.VERSION.SDK_INT >= 30) {
            Insets bars = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
            left = bars.left;
            top = bars.top;
            right = bars.right;
            bottom = bars.bottom;
            ime = insets.getInsets(WindowInsets.Type.ime()).bottom;
        } else {
            // Before API 30 the bottom window inset includes the keyboard; the
            // stable inset is the navigation bar alone.
            left = insets.getSystemWindowInsetLeft();
            top = insets.getSystemWindowInsetTop();
            right = insets.getSystemWindowInsetRight();
            int all = insets.getSystemWindowInsetBottom();
            bottom = Math.min(all, insets.getStableInsetBottom());
            ime = all > bottom ? all : 0;
        }
        v.setPadding(left, top, right, bottom);

        // Surface coordinates: the part of the keyboard above the navigation bar.
        int keyboard = Math.max(0, ime - bottom);
        if (keyboard != mKeyboardHeight) {
            mKeyboardHeight = keyboard;
            int surfaceWidth = Math.max(0, v.getWidth() - left - right);
            int surfaceHeight = Math.max(0, v.getHeight() - top - bottom);
            if (keyboard > 0) {
                QuadNative.surfaceOnKeyboard(true, 0, surfaceHeight - keyboard, surfaceWidth, keyboard);
            } else {
                QuadNative.surfaceOnKeyboard(false, 0, 0, 0, 0);
            }
        }
        return insets;
    }
}

public class MainActivity extends Activity {
    //% MAIN_ACTIVITY_BODY

    private QuadSurface view;

    public static MainActivity sInstance = null;
    public static HiddenEditText sHiddenEditText = null;
    public static String sTextInputText = "";
    public static int sTextInputSelectionStart = 0;
    public static int sTextInputSelectionEnd = 0;
    public static boolean sTextInputIsPassword = false;
    public static boolean sTextInputIsMultiline = false;
    public static boolean sTextInputReadOnly = false;
    // Id of the active text input session (from Rust), 0 when none. Every
    // state report carries it, so Rust drops reports of an earlier session.
    public static long sFocusedElementId = 0;
    // A session started and its text has not arrived yet.
    public static boolean sAwaitingState = false;

    public static String getClipboardTextStatic() {
        if (sInstance != null) {
            return sInstance.getClipboardText();
        }
        return null;
    }

    public static void setClipboardTextStatic(String text) {
        if (sInstance != null) {
            sInstance.setClipboardText(text);
        }
    }

    // Text input session: kind, action, and capitalization codes follow
    // miniquad's TextInputKind, TextInputAction, and TextInputCapitalization.
    public void startTextInput(final int kind, final int action, final boolean autocorrect, final int capitalization, final boolean readOnly, final long session) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (sHiddenEditText == null) return;
                int inputType;
                switch (kind) {
                    case 1: inputType = InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD; break;
                    case 2: inputType = InputType.TYPE_CLASS_NUMBER; break;
                    case 3: inputType = InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL | InputType.TYPE_NUMBER_FLAG_SIGNED; break;
                    case 4: inputType = InputType.TYPE_CLASS_PHONE; break;
                    case 5: inputType = InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS; break;
                    case 6: inputType = InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI; break;
                    case 8: inputType = InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE; break;
                    default: inputType = InputType.TYPE_CLASS_TEXT; break;
                }
                if ((inputType & InputType.TYPE_MASK_CLASS) == InputType.TYPE_CLASS_TEXT && kind != 1) {
                    inputType |= autocorrect ? InputType.TYPE_TEXT_FLAG_AUTO_CORRECT : InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS;
                    switch (capitalization) {
                        case 1: inputType |= InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS; break;
                        case 2: inputType |= InputType.TYPE_TEXT_FLAG_CAP_WORDS; break;
                        case 3: inputType |= InputType.TYPE_TEXT_FLAG_CAP_SENTENCES; break;
                        default: break;
                    }
                }
                int imeAction;
                switch (action) {
                    case 1: imeAction = EditorInfo.IME_ACTION_DONE; break;
                    case 2: imeAction = EditorInfo.IME_ACTION_GO; break;
                    case 3: imeAction = EditorInfo.IME_ACTION_NEXT; break;
                    case 4: imeAction = EditorInfo.IME_ACTION_PREVIOUS; break;
                    case 5: imeAction = EditorInfo.IME_ACTION_SEARCH; break;
                    case 6: imeAction = EditorInfo.IME_ACTION_SEND; break;
                    case 7: imeAction = EditorInfo.IME_ACTION_NONE; break;
                    default:
                        imeAction = kind == 8 ? EditorInfo.IME_ACTION_NONE
                            : kind == 7 ? EditorInfo.IME_ACTION_SEARCH : EditorInfo.IME_ACTION_DONE;
                        break;
                }
                int imeOptions = imeAction | EditorInfo.IME_FLAG_NO_FULLSCREEN;
                if (kind == 1 && Build.VERSION.SDK_INT >= 26) {
                    imeOptions |= EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING;
                }
                finishTextMenu();
                sTextInputIsPassword = kind == 1;
                sTextInputIsMultiline = kind == 8;
                sTextInputReadOnly = readOnly;
                sFocusedElementId = session;
                sAwaitingState = true;
                sHiddenEditText.setIgnoreUpdates(true);
                try {
                    // Forget the previous field's text before the input method
                    // reconnects, so it cannot commit that text into this one.
                    Editable editable = sHiddenEditText.getText();
                    if (editable != null) {
                        android.view.inputmethod.BaseInputConnection.removeComposingSpans(editable);
                        editable.clear();
                    }
                    sTextInputText = "";
                    sHiddenEditText.setInputType(inputType);
                    sHiddenEditText.setImeOptions(imeOptions);
                    sHiddenEditText.setFilters(new android.text.InputFilter[0]);
                } finally {
                    sHiddenEditText.setIgnoreUpdates(false);
                }
                InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
                sHiddenEditText.requestFocus();
                if (imm != null) {
                    imm.restartInput(sHiddenEditText);
                    if (readOnly) {
                        imm.hideSoftInputFromWindow(sHiddenEditText.getWindowToken(), 0);
                    } else {
                        imm.showSoftInput(sHiddenEditText, InputMethodManager.SHOW_IMPLICIT);
                    }
                }
            }
        });
    }

    public void stopTextInput() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                sFocusedElementId = 0;
                sAwaitingState = false;
                finishTextMenu();
                if (sHiddenEditText == null) return;
                Editable editable = sHiddenEditText.getText();
                if (editable != null) {
                    sHiddenEditText.setIgnoreUpdates(true);
                    try {
                        android.view.inputmethod.BaseInputConnection.removeComposingSpans(editable);
                        editable.clear();
                    } finally {
                        sHiddenEditText.setIgnoreUpdates(false);
                    }
                }
                sTextInputText = "";
                sHiddenEditText.clearFocus();
                view.requestFocus();
                InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
                if (imm != null) {
                    imm.hideSoftInputFromWindow(sHiddenEditText.getWindowToken(), 0);
                }
            }
        });
    }

    public void setTextInputState(final String text, final int selectionStart, final int selectionEnd, final int composingStart, final int composingEnd) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (sHiddenEditText == null || sFocusedElementId <= 0) return;
                sTextInputText = text != null ? text : "";
                sAwaitingState = false;
                sHiddenEditText.applyStateFromRust(sTextInputText, selectionStart, selectionEnd, composingStart, composingEnd);
                if (mTextMenu != null) {
                    mTextMenu.invalidate();
                }
            }
        });
    }

    // Floating text toolbar (Cut, Copy, Paste, Select all), the same
    // ActionMode.TYPE_FLOATING toolbar that native text fields and Chrome
    // show. Rust opens it with the selection's bounds in surface pixels.
    private ActionMode mTextMenu = null;
    private final android.graphics.Rect mTextMenuRect = new android.graphics.Rect();

    public void showTextMenu(final int x, final int y, final int width, final int height) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (Build.VERSION.SDK_INT < 23 || view == null || sHiddenEditText == null || sFocusedElementId <= 0) return;
                mTextMenuRect.set(x, y, x + width, y + height);
                if (mTextMenu != null) {
                    mTextMenu.invalidateContentRect();
                    mTextMenu.invalidate();
                    return;
                }
                mTextMenu = view.startActionMode(new TextMenuCallback(), ActionMode.TYPE_FLOATING);
            }
        });
    }

    public void hideTextMenu() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                finishTextMenu();
            }
        });
    }

    // UI thread only.
    void finishTextMenu() {
        if (mTextMenu != null) {
            ActionMode mode = mTextMenu;
            mTextMenu = null;
            mode.finish();
        }
    }

    private class TextMenuCallback extends ActionMode.Callback2 {
        private void populate(Menu menu) {
            menu.clear();
            if (sHiddenEditText == null) return;
            Editable editable = sHiddenEditText.getText();
            int length = editable != null ? editable.length() : 0;
            int start = editable != null ? Selection.getSelectionStart(editable) : 0;
            int end = editable != null ? Selection.getSelectionEnd(editable) : 0;
            boolean selected = start != end;
            boolean editableText = !sTextInputReadOnly;
            boolean secure = sTextInputIsPassword;
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            boolean canPaste = clipboard != null && clipboard.hasPrimaryClip();
            int order = 0;
            if (selected && editableText && !secure) {
                menu.add(Menu.NONE, android.R.id.cut, order++, android.R.string.cut)
                    .setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS);
            }
            if (selected && !secure) {
                menu.add(Menu.NONE, android.R.id.copy, order++, android.R.string.copy)
                    .setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS);
            }
            if (editableText && canPaste) {
                menu.add(Menu.NONE, android.R.id.paste, order++, android.R.string.paste)
                    .setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS);
            }
            if (length > 0 && Math.abs(end - start) < length) {
                menu.add(Menu.NONE, android.R.id.selectAll, order++, android.R.string.selectAll)
                    .setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS);
            }
        }

        @Override
        public boolean onCreateActionMode(ActionMode mode, Menu menu) {
            populate(menu);
            return menu.size() > 0;
        }

        @Override
        public boolean onPrepareActionMode(ActionMode mode, Menu menu) {
            populate(menu);
            return true;
        }

        @Override
        public boolean onActionItemClicked(ActionMode mode, MenuItem item) {
            int id = item.getItemId();
            if (sHiddenEditText == null || sFocusedElementId <= 0) return false;
            // The hidden EditText mirrors the field's text and selection; its
            // edits reach Rust as ordinary editing-state changes.
            sHiddenEditText.onTextContextMenuItem(id);
            if (id == android.R.id.selectAll) {
                mode.invalidate();
            } else {
                finishTextMenu();
            }
            return true;
        }

        @Override
        public void onDestroyActionMode(ActionMode mode) {
            if (mTextMenu == mode) {
                mTextMenu = null;
            }
        }

        @Override
        public void onGetContentRect(ActionMode mode, View v, android.graphics.Rect outRect) {
            outRect.set(mTextMenuRect);
        }
    }

    static {
        System.loadLibrary("LIBRARY_NAME");
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        sInstance = this;
        super.onCreate(savedInstanceState);

        this.requestWindowFeature(Window.FEATURE_NO_TITLE);

        view = new QuadSurface(this);
        sHiddenEditText = new HiddenEditText(this);

        ResizingLayout layout = new ResizingLayout(this);
        layout.addView(view);
        layout.addView(sHiddenEditText);
        setContentView(layout);
        applyWindowMode(false);

        QuadNative.activityOnCreate(this);

        //% MAIN_ACTIVITY_ON_CREATE
    }

    @Override
    protected void onResume() {
        super.onResume();
        QuadNative.activityOnResume();

        //% MAIN_ACTIVITY_ON_RESUME
    }

    // The back button (and back gesture) is delivered to the application as
    // KeyCode::Back. The application closes the activity with
    // miniquad::window::order_quit() when it has nothing to go back to.
    @Override
    public void onBackPressed() {
        QuadNative.surfaceOnKeyDown(KeyEvent.KEYCODE_BACK);
        QuadNative.surfaceOnKeyUp(KeyEvent.KEYCODE_BACK);
    }

    // Haptic kinds follow miniquad's HapticFeedback order.
    public void hapticFeedback(final int kind) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                int constant;
                switch (kind) {
                    case 0: constant = android.view.HapticFeedbackConstants.KEYBOARD_TAP; break;
                    case 1: constant = android.view.HapticFeedbackConstants.VIRTUAL_KEY; break;
                    case 2: constant = android.view.HapticFeedbackConstants.LONG_PRESS; break;
                    case 3: constant = android.view.HapticFeedbackConstants.CLOCK_TICK; break;
                    case 4: constant = Build.VERSION.SDK_INT >= 30
                        ? android.view.HapticFeedbackConstants.CONFIRM
                        : android.view.HapticFeedbackConstants.VIRTUAL_KEY; break;
                    default: constant = Build.VERSION.SDK_INT >= 30
                        ? android.view.HapticFeedbackConstants.REJECT
                        : android.view.HapticFeedbackConstants.LONG_PRESS; break;
                }
                if (view != null) {
                    view.performHapticFeedback(constant);
                }
            }
        });
    }

    public void quitApp() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                finish();
            }
        });
    }

    @Override
    protected void onStop() {
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        QuadNative.activityOnDestroy();
    }

    @Override
    protected void onPause() {
        super.onPause();
        QuadNative.activityOnPause();

        //% MAIN_ACTIVITY_ON_PAUSE
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        //% MAIN_ACTIVITY_ON_ACTIVITY_RESULT
    }

    public void setFullScreen(final boolean fullscreen) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                applyWindowMode(fullscreen);
            }
        });
    }

    // The content always lays out behind the system bars and the keyboard
    // (ResizingLayout pads for the bars itself), so the window never resizes
    // when the keyboard opens. Full screen additionally hides the bars.
    @SuppressWarnings("deprecation")
    private void applyWindowMode(boolean fullscreen) {
        Window window = getWindow();
        window.setSoftInputMode(LayoutParams.SOFT_INPUT_ADJUST_RESIZE | LayoutParams.SOFT_INPUT_STATE_UNSPECIFIED);
        if (Build.VERSION.SDK_INT >= 28) {
            window.getAttributes().layoutInDisplayCutoutMode = fullscreen
                ? LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
                : LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_DEFAULT;
        }
        if (Build.VERSION.SDK_INT >= 30) {
            window.setDecorFitsSystemWindows(false);
            WindowInsetsController controller = window.getInsetsController();
            if (controller != null) {
                if (fullscreen) {
                    controller.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
                    controller.hide(WindowInsets.Type.systemBars());
                } else {
                    controller.show(WindowInsets.Type.systemBars());
                }
            }
        } else {
            int flags = View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION;
            if (fullscreen) {
                flags |= View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                    | View.SYSTEM_UI_FLAG_FULLSCREEN
                    | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY;
            }
            window.getDecorView().setSystemUiVisibility(flags);
        }
    }

    public void showKeyboard(final boolean show) {
        runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (sHiddenEditText == null) return;
                    InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
                    if (imm == null) return;

                    if (show) {
                        sHiddenEditText.requestFocus();
                        imm.showSoftInput(sHiddenEditText, InputMethodManager.SHOW_IMPLICIT);
                    } else {
                        sHiddenEditText.clearFocus();
                        imm.hideSoftInputFromWindow(sHiddenEditText.getWindowToken(), 0);
                    }
                }
            });
    }

    public String getClipboardText() {
        ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);

        if (!clipboard.hasPrimaryClip())
            return null;

        ClipData primaryClip = clipboard.getPrimaryClip();
        if (primaryClip == null || primaryClip.getItemCount() < 1)
            return null;

        CharSequence clipData = clipboard.getPrimaryClip().getItemAt(0).getText();
        if (clipData == null) {
            return null;
        }

        return clipData.toString();
    }
    public void setClipboardText(String text) {
        ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        ClipData clip = ClipData.newPlainText("label", text);
        clipboard.setPrimaryClip(clip);
    }
}

