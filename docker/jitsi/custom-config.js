/**
 * Lumicoria Meet — config.js overrides.
 *
 * Mounted into jitsi/web at /config/custom-config.js. Overrides Jitsi's
 * defaults for room behaviour. Applies to every access path.
 */

// eslint-disable-next-line no-unused-vars
var config = {
    // Neutral defaults — Lumicoria's own lobby handles pre-join UX.
    prejoinPageEnabled:      false,
    prejoinConfig:           { enabled: false },
    enableWelcomePage:       false,
    disableInviteFunctions:  true,
    disableProfile:          false,
    disableShortcuts:        false,
    defaultLanguage:         'en',
    // No "Feedback" prompt after a call — we have our own NPS.
    disableFeedbackMeeting:  true,
    // Kill Jitsi's mobile app upsell banners on mobile web.
    disableThirdPartyRequests: true,
    disableDeepLinking:        true,
    // Analytics + telemetry OFF (we handle our own via Prometheus).
    analytics: {
        rtcstatsEnabled:    false,
        disabled:           true,
    },
};
