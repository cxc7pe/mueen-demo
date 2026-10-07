"use client";

import { Button } from "@/components/ui/button";
import {
  Camera,
  Check,
  Clock3,
  Activity,
  Play,
  Armchair,
  ArrowUp,
  MoveVertical,
  Mic,
} from "lucide-react";

import { useEffect, useRef, useState } from "react";
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

// ---------- Settings ----------
const KNEE_STANDING_ANGLE = 160;
const KNEE_SITTING_ANGLE = 100;
// Hip angle (shoulder-hip-knee) at the TOP of the stand. ~180 = fully upright.
const HIP_FULL_EXTENSION_ANGLE = 155;
// Some forward lean is normal when rising from a chair; only warn above this.
const TORSO_LEAN_MAX_DEGREES = 45;
// After reaching standing, wait this long (taking the best hip angle) before judging.
const STAND_SETTLE_SEC = 0.7;
// Hands-on-thighs check: a wrist counts as "on the thigh" when it is this close to
// the thigh (as a fraction of torso length), away from the hip end of the thigh.
const HAND_ON_THIGH_DISTANCE = 0.3;
const HAND_ON_THIGH_MIN_T = 0.3; // 0 = at the hip, 1 = at the knee
// Warn only if the hand was on the thigh for at least this share of the rise
const HAND_ON_THIGH_MIN_SHARE = 0.5;
const MIN_WRIST_VISIBILITY = 0.5;
const TEST_DURATION = 30;
const SMOOTHING = 0.3;
const FEEDBACK_COOLDOWN_SEC = 4.0;

// "user" = front camera (person sees themselves), "environment" = back camera
const CAMERA_FACING: "user" | "environment" = "user";

// MUST match the installed @mediapipe/tasks-vision version in package.json.
// A mismatch between the JS package and the wasm files breaks Safari/Firefox.
const MEDIAPIPE_VERSION = "1.1.0";
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";


const LEFT_HIP = 23, LEFT_KNEE = 25, LEFT_ANKLE = 27, LEFT_SHOULDER = 11, LEFT_WRIST = 15;
const RIGHT_HIP = 24, RIGHT_KNEE = 26, RIGHT_ANKLE = 28, RIGHT_SHOULDER = 12, RIGHT_WRIST = 16;

// Audio files live in /public/audio/<key>.mp3
const PHRASES = ["intro", "good_rep", "incomplete_stand", "lean_warning", "arms_warning", "time_up"];

// ---------- Helpers ----------
function calcAngle(a: number[], b: number[], c: number[]) {
  const radians = Math.atan2(c[1] - b[1], c[0] - b[0]) - Math.atan2(a[1] - b[1], a[0] - b[0]);
  let angle = Math.abs((radians * 180) / Math.PI);
  if (angle > 180) angle = 360 - angle;
  return angle;
}

function torsoLean(shoulder: number[], hip: number[]) {
  const dx = shoulder[0] - hip[0];
  const dy = shoulder[1] - hip[1];
  return Math.abs((Math.atan2(Math.abs(dx), Math.abs(dy)) * 180) / Math.PI);
}

// Is the wrist resting on the thigh (between mid-thigh and knee)?
// Arms hanging at the sides or crossed on the chest do not count.
function handOnThigh(wrist: number[], hip: number[], knee: number[], torsoLen: number) {
  const tx = knee[0] - hip[0], ty = knee[1] - hip[1];
  const len2 = tx * tx + ty * ty;
  if (len2 < 1e-6) return false;
  const t = ((wrist[0] - hip[0]) * tx + (wrist[1] - hip[1]) * ty) / len2;
  if (t < HAND_ON_THIGH_MIN_T || t > 1.1) return false;
  const px = hip[0] + t * tx, py = hip[1] + t * ty;
  return Math.hypot(wrist[0] - px, wrist[1] - py) / torsoLen < HAND_ON_THIGH_DISTANCE;
}

function checkSitToStand(angle: number) {
  if (angle < KNEE_SITTING_ANGLE) return "SITTING";
  if (angle > KNEE_STANDING_ANGLE) return "STANDING";
  return "TRANSITIONING";
}

// Makes Arabic voice results easier to match (ابدأ / ابدا / إبدأ ...)
function normalizeArabic(t: string) {
  return t
    .replace(/[أإآٱ]/g, "ا")
    .replace(/[ً-ْ]/g, "")
    .trim();
}

// Safari (Mac + every browser on iPhone/iPad, which all use WebKit).
// Continuous speech recognition there is unreliable and, on iPhone, turning the
// mic on routes the voice feedback to the quiet earpiece. So we skip it there.
function isWebKitSafari() {
  if (typeof navigator === "undefined") return false;
  const ua = navigator.userAgent;
  const iOS =
    /iPad|iPhone|iPod/.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  const desktopSafari = /^((?!chrome|chromium|android|crios|fxios|edg).)*safari/i.test(ua);
  return iOS || desktopSafari;
}

async function openCamera(): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("getUserMedia not supported (page must be opened over HTTPS)");
  }
  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: CAMERA_FACING, width: { ideal: 640 }, height: { ideal: 480 } },
    });
  } catch (err) {
    // Some browsers reject the constraints; retry with the simplest request
    if ((err as Error)?.name === "NotAllowedError") throw err;
    return navigator.mediaDevices.getUserMedia({ audio: false, video: true });
  }
}

async function createLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const make = (delegate: "GPU" | "CPU") =>
    PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  try {
    return await make("GPU");
  } catch (err) {
    // GPU (WebGL) can fail on Safari / older phones: fall back to CPU
    console.warn("GPU delegate failed, using CPU:", err);
    return make("CPU");
  }
}

// ---------- Component ----------
export default function MueenCoach() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  // One shared player: once a tap has "unlocked" it, iPhone Safari lets it
  // play any clip later. Separate <audio> elements would each need their own tap.
  const playerRef = useRef<HTMLAudioElement | null>(null);
  const durationsRef = useRef<Record<string, number>>({});
  const lastPlayedRef = useRef<Record<string, number>>({});
  const rafRef = useRef<number>(0);
  const readyRef = useRef(false);
  const audioUnlockedRef = useRef(false);
  const lastVideoTimeRef = useRef(-1);
  // The mic is switched off while Mueen talks / during the test. Leaving it on
  // makes phones duck or stutter the voice, and it can hear its own intro.
  const pauseMicRef = useRef<() => void>(() => {});
  const resumeMicRef = useRef<() => void>(() => {});
  const resumeMicTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const stateRef = useRef({
    smoothedKnee: null as number | null,
    stage: null as string | null,
    reps: 0,
    testStart: null as number | null,
    testDone: false,
    maxLean: 0,
    riseFrames: 0,
    handOnThighFrames: 0,
    // Feedback for the current rep is decided a moment after standing up
    judgeAt: null as number | null,
    maxHipTop: 0,
  });

  const [reps, setReps] = useState(0);
  const [label, setLabel] = useState("");
  const [timeLeft, setTimeLeft] = useState<number | null>(null);
  const [ready, setReady] = useState(false);
  const [cameraError, setCameraError] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(false);

  // Must run inside a tap/click. Plays a clip silently to unlock audio on mobile.
  function unlockAudio() {
    const player = playerRef.current;
    // Skip if already unlocked, or if a phrase (e.g. the intro) is already playing
    if (!player || audioUnlockedRef.current || !player.paused) return;
    player.muted = true;
    player.src = "/audio/good_rep.mp3";
    player
      .play()
      .then(() => {
        audioUnlockedRef.current = true;
        // Only reset if a real phrase hasn't taken over the player meanwhile
        if (player.muted) {
          player.pause();
          player.currentTime = 0;
          player.muted = false;
        }
      })
      .catch(() => {
        player.muted = false; // try again on the next tap
      });
  }

  function playPhrase(key: string, force = false, cooldown = FEEDBACK_COOLDOWN_SEC) {
    const player = playerRef.current;
    if (!player) return;
    const now = performance.now() / 1000;
    const last = lastPlayedRef.current[key] || 0;
    if (!force && now - last < cooldown) return;

    player.pause();
    player.muted = false;
    player.src = `/audio/${key}.mp3`;
    player.currentTime = 0;
    player
      .play()
      .then(() => {
        audioUnlockedRef.current = true;
      })
      .catch((e) => console.warn("Audio blocked:", key, e?.name));
    lastPlayedRef.current[key] = now;
  }

  // Called by the button and by the voice command
  function startTest() {
    if (!readyRef.current) return;
    const s = stateRef.current;
    clearTimeout(resumeMicTimerRef.current);
    pauseMicRef.current();
    playPhrase("intro", true); // inside the tap, so this also unlocks audio
    const introDuration = durationsRef.current["intro"] || 5;
    s.testStart = performance.now() / 1000 + introDuration + 1.0;
    s.testDone = false;
    s.reps = 0;
    s.stage = null;
    s.judgeAt = null;
    setReps(0);
    setTimeLeft(null);
  }

  function processFrame() {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    const canvas = canvasRef.current;
    if (!video || !landmarker || !canvas) return;
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const s = stateRef.current;
    const now = performance.now() / 1000;
    const w = canvas.width, h = canvas.height;

    // Only run the model on a new camera frame (saves battery on phones)
    if (video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;
      const result = landmarker.detectForVideo(video, performance.now());
      const lm = result.landmarks?.[0];

      if (lm && lm.length > RIGHT_ANKLE) {
        const vis = (i: number) => lm[i]?.visibility ?? 0;
        const pt = (i: number) => [lm[i].x * w, lm[i].y * h];

        const leftVis = (vis(LEFT_HIP) + vis(LEFT_KNEE) + vis(LEFT_ANKLE)) / 3;
        const rightVis = (vis(RIGHT_HIP) + vis(RIGHT_KNEE) + vis(RIGHT_ANKLE)) / 3;
        const useLeft = leftVis >= rightVis;

        const hip = pt(useLeft ? LEFT_HIP : RIGHT_HIP);
        const knee = pt(useLeft ? LEFT_KNEE : RIGHT_KNEE);
        const ankle = pt(useLeft ? LEFT_ANKLE : RIGHT_ANKLE);
        const shoulder = pt(useLeft ? LEFT_SHOULDER : RIGHT_SHOULDER);
        const wrist = pt(useLeft ? LEFT_WRIST : RIGHT_WRIST);

        const kneeAngle = calcAngle(hip, knee, ankle);
        const hipAngle = calcAngle(shoulder, hip, knee);
        const lean = torsoLean(shoulder, hip);
        const torsoLen = Math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1]) + 1e-6;
        // Either hand resting on its own thigh (only if the camera can see that wrist)
        const handPush =
          (vis(LEFT_WRIST) >= MIN_WRIST_VISIBILITY &&
            handOnThigh(pt(LEFT_WRIST), pt(LEFT_HIP), pt(LEFT_KNEE), torsoLen)) ||
          (vis(RIGHT_WRIST) >= MIN_WRIST_VISIBILITY &&
            handOnThigh(pt(RIGHT_WRIST), pt(RIGHT_HIP), pt(RIGHT_KNEE), torsoLen));

        if (Number.isFinite(kneeAngle)) {
          s.smoothedKnee =
            s.smoothedKnee == null ? kneeAngle : SMOOTHING * kneeAngle + (1 - SMOOTHING) * s.smoothedKnee;
        }

        if (s.smoothedKnee != null) {
          const state = checkSitToStand(s.smoothedKnee);
          const counting = s.testStart != null && !s.testDone && now >= s.testStart;

          // While rising out of the chair: track forward lean and arm use
          if (s.stage === "SITTING" && state !== "SITTING") {
            s.maxLean = Math.max(s.maxLean, lean);
            s.riseFrames += 1;
            if (handPush) s.handOnThighFrames += 1;
          }

          // After standing up: track how straight the hips get at the top
          if (s.judgeAt != null && state === "STANDING") {
            s.maxHipTop = Math.max(s.maxHipTop, hipAngle);
          }

          // Give feedback once they've settled at the top (or started sitting again)
          if (s.judgeAt != null && (now >= s.judgeAt || state !== "STANDING")) {
            s.judgeAt = null;
            const issues: string[] = [];
            if (s.maxHipTop < HIP_FULL_EXTENSION_ANGLE) issues.push("incomplete_stand");
            if (s.maxLean > TORSO_LEAN_MAX_DEGREES) issues.push("lean_warning");
            if (s.riseFrames >= 3 && s.handOnThighFrames / s.riseFrames >= HAND_ON_THIGH_MIN_SHARE) {
              issues.push("arms_warning");
            }
            playPhrase(issues[0] || "good_rep");
          }

          if (state === "SITTING") {
            s.stage = "SITTING";
            s.maxLean = 0;
            s.riseFrames = 0;
            s.handOnThighFrames = 0;
          } else if (state === "STANDING" && s.stage === "SITTING") {
            s.stage = "STANDING";
            if (counting) {
              s.reps += 1;
              setReps(s.reps);
              s.maxHipTop = hipAngle;
              s.judgeAt = now + STAND_SETTLE_SEC;
            }
          }
          setLabel(state);
        }

        ctx.fillStyle = "lime";
        [hip, knee, ankle, shoulder, wrist].forEach(([x, y]) => {
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI);
          ctx.fill();
        });
      } else {
        s.smoothedKnee = null;
        setLabel("");
      }
    }

    // ----- Timer -----
    if (s.testStart != null && !s.testDone) {
      const elapsed = now - s.testStart;
      setTimeLeft(Math.ceil(Math.max(0, TEST_DURATION - elapsed)));
      if (elapsed >= TEST_DURATION) {
        s.testDone = true;
        setTimeLeft(0);
        playPhrase("time_up", true);
        // Turn the mic back on after the closing message has finished
        const endDelay = ((durationsRef.current["time_up"] || 3) + 0.8) * 1000;
        clearTimeout(resumeMicTimerRef.current);
        resumeMicTimerRef.current = setTimeout(() => resumeMicRef.current(), endDelay);
      }
    }
  }

  function loop() {
    // One bad frame must never stop the whole app
    try {
      processFrame();
    } catch (err) {
      console.warn("Frame skipped:", err);
    }
    rafRef.current = requestAnimationFrame(loop);
  }

  // Camera + pose model setup
  useEffect(() => {
    let cancelled = false;
    let stream: MediaStream | null = null;

    const player = new Audio();
    player.preload = "auto";
    player.setAttribute("playsinline", "");
    playerRef.current = player;

    // Warm the cache and read clip lengths (needed for the intro delay)
    PHRASES.forEach((key) => {
      const a = new Audio();
      a.preload = "auto";
      a.addEventListener("loadedmetadata", () => {
        if (Number.isFinite(a.duration)) durationsRef.current[key] = a.duration;
      });
      a.src = `/audio/${key}.mp3`;
      a.load();
    });

    async function init() {
      try {
        // Camera first so the permission prompt appears right away
        stream = await openCamera();
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        const video = videoRef.current;
        if (video) {
          video.muted = true;
          video.playsInline = true;
          video.srcObject = stream;
          try {
            await video.play();
          } catch (err) {
            console.warn("video.play() was refused, will retry on tap:", err);
          }
        }

        const landmarker = await createLandmarker();
        if (cancelled) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;

        readyRef.current = true;
        setReady(true);
        rafRef.current = requestAnimationFrame(loop);
      } catch (err) {
        console.error("Init failed:", err);
        if (!cancelled) setCameraError(true);
      }
    }
    init();

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
      stream?.getTracks().forEach((t) => t.stop());
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
      player.pause();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Voice command: "ابدأ" / "توقف" (Chrome / Edge / Android only)
  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const w = window as any;
    const SpeechRecognition = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!SpeechRecognition || isWebKitSafari()) return;

    let stopped = false;
    let paused = false;
    let failures = 0;
    let restartTimer: ReturnType<typeof setTimeout> | undefined;
    const recognition = new SpeechRecognition();
    recognition.lang = "ar-SA";
    recognition.continuous = true;
    recognition.interimResults = false;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onresult = (event: any) => {
      failures = 0;
      if (!readyRef.current) return; // ignore commands until the camera/model are ready
      const raw = event.results[event.results.length - 1][0].transcript;
      const text = normalizeArabic(raw);
      const st = stateRef.current;
      const testRunning = st.testStart != null && !st.testDone;
      if (text.includes("ابدا") && !testRunning) {
        // Release the mic first, then speak once the phone has switched audio back
        pauseMicRef.current();
        setTimeout(startTest, 400);
      } else if (text.includes("توقف")) {
        st.testDone = true;
      }
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onerror = (e: any) => {
      if (["not-allowed", "service-not-allowed", "audio-capture", "language-not-supported"].includes(e.error)) {
        stopped = true; // permanent problem: stop retrying, the button still works
        setVoiceSupported(false);
      } else if (e.error !== "no-speech" && e.error !== "aborted") {
        failures += 1;
      }
    };

    recognition.onend = () => {
      if (stopped || paused || failures > 5) return;
      // Small delay avoids a tight restart loop on Android
      restartTimer = setTimeout(() => {
        try {
          recognition.start();
        } catch {
          /* already started */
        }
      }, 300);
    };

    recognition.onstart = () => setVoiceSupported(true);

    pauseMicRef.current = () => {
      paused = true;
      clearTimeout(restartTimer);
      try {
        recognition.abort();
      } catch {
        /* ignore */
      }
    };
    resumeMicRef.current = () => {
      if (stopped || !paused) return;
      paused = false;
      failures = 0;
      try {
        recognition.start();
      } catch {
        /* already started */
      }
    };

    try {
      recognition.start();
    } catch {
      /* ignore */
    }

    return () => {
      stopped = true;
      clearTimeout(restartTimer);
      clearTimeout(resumeMicTimerRef.current);
      pauseMicRef.current = () => {};
      resumeMicRef.current = () => {};
      recognition.onend = null;
      try {
        recognition.stop();
      } catch {
        /* ignore */
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleUserGesture() {
    unlockAudio();
    // Safari sometimes refuses the first video.play(); a tap fixes it
    const v = videoRef.current;
    if (v && v.paused && v.srcObject) v.play().catch(() => {});
  }

  return (
    <div
      dir="rtl"
      lang="ar"
      onClick={handleUserGesture}
      onTouchEnd={handleUserGesture}
      className="min-h-screen bg-wellness font-ar text-foreground"
    >
      <div className="mx-auto max-w-[1400px] px-5 py-7 sm:px-8 lg:px-12 lg:py-10">
        <header className="flex flex-wrap items-center justify-between gap-5 border-b border-border/70 pb-7">
          <div className="flex items-center gap-4">
            <div className="grid size-16 shrink-0 place-items-center rounded-[20px] bg-primary text-primary-foreground shadow-action" aria-hidden="true"><span className="text-4xl font-extrabold">م</span></div>
            <div><h1 className="text-4xl font-extrabold leading-tight">معين</h1><p className="mt-1 text-lg font-medium text-muted-foreground">معك، في كل حركة</p></div>
          </div>
        </header>
        <div className="flex flex-wrap items-end justify-between gap-4 pb-7 pt-8">
          <div><p className="mb-2 text-lg font-semibold text-primary">وقت للحركة، وقت لك</p><h2 className="text-3xl font-extrabold leading-normal sm:text-4xl">تمرين الجلوس والوقوف</h2></div>
          <div className="flex items-center gap-2 text-lg font-semibold text-muted-foreground"><Clock3 className="size-5" />تمرين لمدة ٣٠ ثانية</div>
        </div>
        <main className="grid items-stretch gap-6 lg:grid-cols-[1.7fr_1fr]">
          <section className="min-w-0" aria-label="الكاميرا ووضعية الجسم">
            <div className="relative aspect-[4/3] overflow-hidden rounded-3xl border border-surface-edge bg-secondary shadow-frost">
              {/* Not display:none — Safari stops decoding hidden videos. It sits invisibly under the canvas. */}
              <video
                ref={videoRef}
                className="pointer-events-none absolute inset-0 h-full w-full opacity-0"
                autoPlay
                playsInline
                muted
                aria-hidden="true"
              />
              {!ready && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 pb-24 text-primary" aria-hidden="true">
                  <div className="grid size-24 place-items-center rounded-full bg-card/80 shadow-soft">
                    <Armchair className="size-12" />
                  </div>
                  <p className="text-xl font-bold text-foreground">تمرين الجلوس والوقوف</p>
                </div>
              )}
              <canvas ref={canvasRef} className={ready ? "absolute inset-0 h-full w-full object-contain" : "absolute inset-0 h-full w-full opacity-0"} aria-label="عرض الكاميرا المباشر مع نقاط تتبّع الحركة" />
              <div className="absolute right-4 top-4 flex items-center gap-2 rounded-full bg-card/95 px-4 py-2 text-base font-bold text-foreground shadow-soft sm:right-5 sm:top-5"><Camera className="size-5 text-primary" />{ready ? "الكاميرا المباشرة" : "الكاميرا"}</div>
              {!ready && !cameraError && <div className="absolute bottom-5 left-5 right-5 rounded-2xl border border-surface-edge bg-card/95 px-5 py-4 shadow-soft"><p className="text-lg font-bold text-foreground">بانتظار تفعيل الكاميرا</p><p className="mt-1 text-base text-muted-foreground">اسمح للمتصفح باستخدام الكاميرا لبدء التمرين.</p></div>}
              {cameraError && <div className="absolute bottom-5 left-5 right-5 rounded-2xl border border-amber-edge bg-warmsoft px-5 py-4 shadow-soft"><p className="text-lg font-bold text-amber-deep">تعذّر تشغيل الكاميرا</p><p className="mt-1 text-base text-amber-deep">تأكد من السماح باستخدام الكاميرا ثم أعد تحميل الصفحة.</p></div>}
            </div>
            <div className="mt-4 flex min-h-24 flex-wrap items-center justify-between gap-4 rounded-2xl border border-surface-edge bg-card/80 px-6 py-4 shadow-soft">
              <div className="flex items-center gap-4"><div className="grid size-12 shrink-0 place-items-center rounded-full bg-secondary text-primary" aria-hidden="true">{label === "SITTING" ? <Armchair className="size-6" /> : label === "STANDING" ? <ArrowUp className="size-6" /> : <MoveVertical className="size-6" />}</div><div><p className="text-base font-semibold text-muted-foreground">وضعيتك الآن</p><p className="min-h-10 text-2xl font-extrabold text-brand-deep" aria-live="polite">{label === "SITTING" ? "جالس" : label === "STANDING" ? "واقف" : label === "TRANSITIONING" ? "في حركة" : "بانتظار الحركة"}</p></div></div>
              {label && <span className="flex items-center gap-2 text-base font-semibold text-primary"><Check className="size-5" />تم رصد الحركة</span>}
            </div>
          </section>
          <aside className="flex min-w-0 flex-col gap-5" aria-label="نتائج التمرين">
            <div className="rounded-3xl border border-surface-edge bg-card/75 p-6 shadow-frost sm:p-7">
              <div className="flex items-center justify-between gap-3"><h3 className="text-xl font-bold">عدد التكرارات</h3><Activity className="size-6 text-primary" /></div>
              <div className="mt-5 flex min-h-24 items-baseline gap-4"><span className="font-num text-7xl font-semibold leading-none text-brand-deep" aria-live="polite">{reps.toLocaleString("ar-SA")}</span><span className="text-lg font-semibold text-muted-foreground">تكرار</span></div>
              <div className="mt-4 h-1.5 rounded-full bg-secondary" />
            </div>
            <div className="rounded-3xl border border-amber-edge bg-warmsoft p-6 shadow-frost sm:p-7">
              <div className="flex items-center justify-between gap-3"><h3 className="text-xl font-bold">الوقت المتبقي</h3><Clock3 className="size-6 text-amber-deep" /></div>
              <div className="mt-5 flex min-h-24 items-baseline gap-4"><span className="font-num text-7xl font-semibold leading-none text-amber-deep" role="timer">{(timeLeft == null ? TEST_DURATION : Math.min(TEST_DURATION, timeLeft)).toLocaleString("ar-SA")}</span><span className="text-lg font-semibold text-amber-deep">ثانية</span></div>
              <p className="mt-4 text-base font-semibold text-amber-deep">{timeLeft === 0 ? "اكتمل التمرين، أحسنت!" : timeLeft == null ? "على مهل، ابدأ حين تكون مستعدًا" : timeLeft > TEST_DURATION ? "استعدّ، سيبدأ العدّ بعد التوجيه" : "خذ وقتك، كل حركة تُحسب"}</p>
            </div>
            <div className="mt-auto flex items-center gap-3 rounded-2xl border border-primary/25 bg-secondary px-6 py-4 text-lg font-semibold text-primary"><Mic className="size-5 shrink-0" />{voiceSupported ? "اضغط زر البدء، أو قل «ابدأ» ليبدأ التمرين" : "اضغط زر البدء ليبدأ التمرين"}</div>
            <Button variant="exercise" size="exercise" onClick={startTest} disabled={!ready} className="mt-3"><Play className="fill-current" />{timeLeft === 0 ? "ابدأ التمرين من جديد" : "ابدأ التمرين"}</Button>
          </aside>
        </main>
        <footer className="mt-9 flex flex-wrap items-center justify-between gap-3 border-t border-border/70 pt-5 text-sm text-muted-foreground"><span>معين · رفيق الحركة</span><span>بخطوات هادئة، نحو نشاط أفضل</span></footer>
      </div>
    </div>
  );
}
