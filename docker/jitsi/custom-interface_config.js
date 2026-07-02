/**
 * Lumicoria Meet — interface_config.js overrides.
 *
 * Mounted at /config/custom-interface_config.js. Merges into the
 * container's generated interface_config.js. Applies to BOTH direct URL
 * access and iframe access via JitsiEmbed. Per-org branding still comes
 * from the React layer via interfaceConfigOverwrite; this file is the
 * fallback that kills Jitsi's own logos + welcome page.
 *
 * IMPORTANT: this file must MERGE into the existing `interfaceConfig`
 * object, NOT replace it. Replacing wipes out toolbar defaults and
 * other config the app depends on.
 *
 * See https://github.com/jitsi/jitsi-meet/blob/master/interface_config.js
 * for the full list of overridable keys.
 */

/* global interfaceConfig */

// ── Names + brand-mark ────────────────────────────────────────────
interfaceConfig.APP_NAME        = 'Lumicoria Meet';
interfaceConfig.NATIVE_APP_NAME = 'Lumicoria Meet';
interfaceConfig.PROVIDER_NAME   = 'Lumicoria';

// Kill every Jitsi watermark.
interfaceConfig.SHOW_JITSI_WATERMARK        = false;
interfaceConfig.SHOW_WATERMARK_FOR_GUESTS   = false;
interfaceConfig.JITSI_WATERMARK_LINK        = 'https://lumicoria.ai';
interfaceConfig.SHOW_POWERED_BY             = false;
interfaceConfig.SHOW_BRAND_WATERMARK        = false;
interfaceConfig.BRAND_WATERMARK_LINK        = 'https://lumicoria.ai';
interfaceConfig.SHOW_PROMOTIONAL_CLOSE_PAGE = false;

// Kill the mobile-app promo on mobile browsers.
interfaceConfig.MOBILE_APP_PROMO = false;

// ── Welcome page (the empty landing before you type a room) ───────
interfaceConfig.DEFAULT_WELCOME_PAGE_LOGO_URL = 'https://lumicoria.ai/logo-wide.png';
interfaceConfig.DISPLAY_WELCOME_PAGE_CONTENT  = false;
interfaceConfig.DISPLAY_WELCOME_FOOTER        = false;
interfaceConfig.HIDE_INVITE_MORE_HEADER       = true;

// ── Colors ────────────────────────────────────────────────────────
interfaceConfig.DEFAULT_BACKGROUND          = '#0F172A';
interfaceConfig.DEFAULT_REMOTE_DISPLAY_NAME = 'Participant';
interfaceConfig.DEFAULT_LOCAL_DISPLAY_NAME  = 'You';

// ── Legal + support links point at Lumicoria ─────────────────────
interfaceConfig.SUPPORT_URL              = 'https://lumicoria.ai/support';
interfaceConfig.PRIVACY_POLICY_URL       = 'https://lumicoria.ai/privacy';
interfaceConfig.TERMS_URL                = 'https://lumicoria.ai/terms';
interfaceConfig.LIVE_STREAMING_HELP_LINK = 'https://lumicoria.ai/docs/livestreaming';
