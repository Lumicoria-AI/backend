/**
 * Lumicoria Meet — interface_config.js overrides.
 *
 * Mounted into jitsi/web at /config/custom-interface_config.js.
 * Everything here overrides the container's defaults — takes effect for
 * BOTH direct URL access (https://meet.lumicoria.ai/...) and iframe
 * access via JitsiEmbed. Per-org branding still comes from the React
 * layer via interfaceConfigOverwrite; this file is the fallback that
 * kills Jitsi's own logos + welcome page.
 *
 * See https://github.com/jitsi/jitsi-meet/blob/master/interface_config.js
 * for the full list of overridable keys.
 */

// eslint-disable-next-line no-unused-vars
var interfaceConfig = {
    // ── Names + brand-mark ────────────────────────────────────────────
    APP_NAME:          'Lumicoria Meet',
    NATIVE_APP_NAME:   'Lumicoria Meet',
    PROVIDER_NAME:     'Lumicoria',

    // Kill every Jitsi watermark. The React layer adds our own via
    // DEFAULT_LOGO_URL when an org configures a logo.
    SHOW_JITSI_WATERMARK:      false,
    SHOW_WATERMARK_FOR_GUESTS: false,
    JITSI_WATERMARK_LINK:      'https://lumicoria.ai',
    SHOW_POWERED_BY:           false,
    SHOW_BRAND_WATERMARK:      false,
    BRAND_WATERMARK_LINK:      'https://lumicoria.ai',
    SHOW_PROMOTIONAL_CLOSE_PAGE: false,

    // Kill the mobile-app promo that appears on mobile browsers.
    MOBILE_APP_PROMO: false,

    // ── Welcome page (the empty landing before you type a room) ───────
    // Hide it entirely — Lumicoria's own dashboard is the landing.
    DEFAULT_WELCOME_PAGE_LOGO_URL: 'https://lumicoria.ai/logo-wide.png',
    DISPLAY_WELCOME_PAGE_CONTENT:  false,
    DISPLAY_WELCOME_FOOTER:        false,
    HIDE_INVITE_MORE_HEADER:       true,
    DISABLE_FOCUS_INDICATOR:       false,

    // ── Colors ────────────────────────────────────────────────────────
    // Fallback backgrounds. React layer overrides via CSS custom
    // properties for per-org primary/accent colors.
    DEFAULT_BACKGROUND:           '#0F172A',
    DEFAULT_REMOTE_DISPLAY_NAME:  'Participant',
    DEFAULT_LOCAL_DISPLAY_NAME:   'You',

    // ── Toolbar ───────────────────────────────────────────────────────
    // Default toolbar for direct-URL access. React layer sends its own
    // host-aware list when embedding via iframe.
    TOOLBAR_BUTTONS: [
        'microphone', 'camera', 'desktop', 'fullscreen',
        'fodeviceselection', 'hangup', 'chat', 'raisehand',
        'videoquality', 'filmstrip', 'settings', 'tileview',
        'select-background', 'participants-pane', 'videobackgroundblur',
    ],

    // ── Miscellaneous ─────────────────────────────────────────────────
    // No feedback thumbs-up/down after a call — we have our own NPS.
    FILM_STRIP_MAX_HEIGHT: 120,
    VERTICAL_FILMSTRIP:    true,
    // Turn off the "About" / "Terms" links that point back at Jitsi.
    LANG_DETECTION:           true,
    LIVE_STREAMING_HELP_LINK: 'https://lumicoria.ai/docs/livestreaming',
    SUPPORT_URL:              'https://lumicoria.ai/support',
    PRIVACY_POLICY_URL:       'https://lumicoria.ai/privacy',
    TERMS_URL:                'https://lumicoria.ai/terms',

    // Hide the Jitsi "Powered by" strip at the bottom of the pre-join
    // page even when we can't disable pre-join entirely.
    DISABLE_JOIN_LEAVE_NOTIFICATIONS: false,
    DISABLE_PRESENCE_STATUS:          false,
    DISABLE_RINGING:                  false,
};
