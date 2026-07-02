/**
 * Lumicoria Meet — config.js overrides.
 *
 * Mounted at /config/custom-config.js. IMPORTANT: this file runs AFTER
 * the container's generated config.js. It must MERGE properties into
 * the existing `config` object, NOT replace it — replacing wipes out
 * hosts.domain / bosh / websocket / etc. which the meeting depends on.
 *
 * Ref: https://github.com/jitsi/docker-jitsi-meet/blob/master/web/rootfs/defaults/config.js
 */

/* global config */

// Show the prejoin screen so users can preview camera/mic + name themselves
// before entering. The Lumicoria embed can still override this per-huddle
// via configOverwrite if we want to skip it inside the app iframe.
config.prejoinPageEnabled     = true;
config.prejoinConfig          = { enabled: true };
config.enableWelcomePage      = false;
config.disableInviteFunctions = true;
config.disableProfile         = false;
config.disableShortcuts       = false;
config.defaultLanguage        = 'en';

// No "Feedback" prompt after a call — we have our own NPS.
config.disableFeedbackMeeting = true;

// Kill deep-linking to the Jitsi mobile app on mobile web.
config.disableDeepLinking = true;

// Camera + mic behaviour.
config.startWithAudioMuted = false;
config.startWithVideoMuted = false;

// Enable P2P for direct-URL calls with only two users (avoids relaying
// through JVB when both peers can talk directly).
config.p2p = Object.assign({}, config.p2p || {}, { enabled: true });
