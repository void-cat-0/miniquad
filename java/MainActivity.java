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
                QuadNative.surfaceOnImeAction(actionId);
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
        if (MainActivity.sFocusedElementId == -1 || MainActivity.sFocusedElementId == 0) return;

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

    private float mDownX = 0;
    private float mDownY = 0;
    private boolean mIsLongPress = false;
    private Runnable mLongPressRunnable = new Runnable() {
        @Override
        public void run() {
            if (MainActivity.sFocusedElementId != 0 && MainActivity.sFocusedElementId != -1 && MainActivity.sHiddenEditText != null) {
                mIsLongPress = true;
                final float x = mDownX;
                final float y = mDownY;
                MainActivity.sHiddenEditText.post(new Runnable() {
                    @Override
                    public void run() {
                        if (MainActivity.sHiddenEditText != null) {
                            MainActivity.sHiddenEditText.requestFocus();
                            if (android.os.Build.VERSION.SDK_INT >= 24) {
                                MainActivity.sHiddenEditText.showContextMenu(x, y);
                            } else {
                                MainActivity.sHiddenEditText.showContextMenu();
                            }
                        }
                    }
                });
            }
        }
    };

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
            mDownX = x;
            mDownY = y;
            mIsLongPress = false;
            if (getHandler() != null) {
                getHandler().removeCallbacks(mLongPressRunnable);
                getHandler().postDelayed(mLongPressRunnable, 500);
            }
            QuadNative.surfaceOnTouch(id, 0, x, y, event.getEventTime());

            break;
        }
        case MotionEvent.ACTION_MOVE: {
            for (int i = 0; i < pointerCount; i++) {
                final int id = event.getPointerId(i);
                final float x = event.getX(i);
                final float y = event.getY(i);
                if (Math.hypot(x - mDownX, y - mDownY) > 25 && getHandler() != null) {
                    getHandler().removeCallbacks(mLongPressRunnable);
                }
                QuadNative.surfaceOnTouch(id, 1, x, y, event.getEventTime());
            }
            break;
        }
        case MotionEvent.ACTION_UP:
        case MotionEvent.ACTION_POINTER_UP: {
            if (getHandler() != null) {
                getHandler().removeCallbacks(mLongPressRunnable);
            }
            int pointerIndex = event.getActionIndex();
            int id = event.getPointerId(pointerIndex);
            final float x = event.getX(pointerIndex);
            final float y = event.getY(pointerIndex);
            if (!mIsLongPress && MainActivity.sHiddenEditText != null && MainActivity.sFocusedElementId != 0 && MainActivity.sFocusedElementId != -1) {
                InputMethodManager imm = (InputMethodManager) getContext().getSystemService(Context.INPUT_METHOD_SERVICE);
                if (imm != null) {
                    imm.viewClicked(MainActivity.sHiddenEditText);
                }
            }
            int phase = mIsLongPress ? 3 : 2;
            QuadNative.surfaceOnTouch(id, phase, x, y, event.getEventTime());
            break;
        }
        case MotionEvent.ACTION_CANCEL: {
            if (getHandler() != null) {
                getHandler().removeCallbacks(mLongPressRunnable);
            }
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
        if (MainActivity.sFocusedElementId != -1 && MainActivity.sFocusedElementId != 0) {
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
        if (MainActivity.sHiddenEditText != null && MainActivity.sFocusedElementId != -1 && MainActivity.sFocusedElementId != 0) {
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

    private boolean mKeyboardVisible = false;
    private int mKeyboardHeight = 0;

    @Override
    @SuppressWarnings("deprecation")
    public WindowInsets onApplyWindowInsets(View v, WindowInsets insets) {
        int left, top, right, bottom, keyboard;
        if (Build.VERSION.SDK_INT >= 30) {
            Insets imeInsets = insets.getInsets(WindowInsets.Type.ime());
            Insets sysInsets = insets.getInsets(WindowInsets.Type.systemBars());
            left = sysInsets.left;
            top = sysInsets.top;
            right = sysInsets.right;
            keyboard = imeInsets.bottom > sysInsets.bottom ? imeInsets.bottom : 0;
            bottom = keyboard > 0 ? keyboard : sysInsets.bottom;
        } else {
            // Before API 30 the bottom system inset includes the keyboard;
            // anything taller than a navigation bar is taken as a keyboard.
            left = insets.getSystemWindowInsetLeft();
            top = insets.getSystemWindowInsetTop();
            right = insets.getSystemWindowInsetRight();
            bottom = insets.getSystemWindowInsetBottom();
            keyboard = bottom > v.getHeight() / 6 ? bottom : 0;
        }
        v.setPadding(left, top, right, bottom);

        // The surface ends at the keyboard's top edge, so in surface
        // coordinates the keyboard lies just below the surface.
        boolean visible = keyboard > 0;
        if (visible != mKeyboardVisible || keyboard != mKeyboardHeight) {
            mKeyboardVisible = visible;
            mKeyboardHeight = keyboard;
            int surfaceWidth = Math.max(0, v.getWidth() - left - right);
            int surfaceHeight = Math.max(0, v.getHeight() - top - bottom);
            if (visible) {
                QuadNative.surfaceOnKeyboard(true, 0, surfaceHeight, surfaceWidth, keyboard);
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
    public static long sFocusedElementId = -1;

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
    public void startTextInput(final int kind, final int action, final boolean autocorrect, final int capitalization, final boolean readOnly) {
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
                sTextInputIsPassword = kind == 1;
                sTextInputIsMultiline = kind == 8;
                sFocusedElementId = 1;
                sHiddenEditText.setIgnoreUpdates(true);
                try {
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
                if (sHiddenEditText == null) return;
                Editable editable = sHiddenEditText.getText();
                if (editable != null) {
                    android.view.inputmethod.BaseInputConnection.removeComposingSpans(editable);
                }
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
                if (sHiddenEditText == null || sFocusedElementId != 1) return;
                sTextInputText = text != null ? text : "";
                sHiddenEditText.applyStateFromRust(sTextInputText, selectionStart, selectionEnd, composingStart, composingEnd);
            }
        });
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
                    View decorView = getWindow().getDecorView();

                    if (fullscreen) {
                        getWindow().setFlags(LayoutParams.FLAG_LAYOUT_NO_LIMITS, LayoutParams.FLAG_LAYOUT_NO_LIMITS);
                        if (Build.VERSION.SDK_INT >= 28) {
                            getWindow().getAttributes().layoutInDisplayCutoutMode = LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
                        }
                        if (Build.VERSION.SDK_INT >= 30) {
                            getWindow().setDecorFitsSystemWindows(false);
                        } else {
                            int uiOptions = View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY;
                            decorView.setSystemUiVisibility(uiOptions);
                        }
                    }
                    else {
                        if (Build.VERSION.SDK_INT >= 30) {
                            getWindow().setDecorFitsSystemWindows(true);
                        } else {
                          decorView.setSystemUiVisibility(0);
                        }

                    }
                }
            });
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

