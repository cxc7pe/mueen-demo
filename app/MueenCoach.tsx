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
  AlertTriangle,
} from "lucide-react";

import { useEffect, useRef, useState } from "react";
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

// ---------- Settings ----------
const KNEE_STANDING_ANGLE = 160;
const KNEE_SITTING_ANGLE = 100;
const HIP_FULL_EXTENSION_ANGLE = 160;
const TORSO_LEAN_MAX_DEGREES = 30;
const ARM_PUSH_DISTANCE_RATIO = 0.35;
const TEST_DURATION = 30;
const SMOOTHING = 0.3;
const FEEDBACK_COOLDOWN_SEC = 4.0;

// "user" = front camera (person sees themselves), "environment" = back camera
const CAMERA_FACING: "user" | "environment" = "user";

// Body-in-frame check
const MIN_VISIBILITY = 0.5; // landmark confidence needed to count as visible
const FRAME_MARGIN = 0.02; // landmark must be inside the frame by this margin
const LOST_AFTER_SEC = 0.8; // body must be missing this long before we warn
const OK_AFTER_SEC = 1.0; // body must be fully visible this long before starting
const STEP_BACK_COOLDOWN_SEC = 7;

const exercisePreview = "/exercise-preview.jpg";

const LEFT_HIP = 23, LEFT_KNEE = 25, LEFT_ANKLE = 27, LEFT_SHOULDER = 11, LEFT_WRIST = 15;
const RIGHT_HIP = 24, RIGHT_KNEE = 26, RIGHT_ANKLE = 28, RIGHT_SHOULDER = 12, RIGHT_WRIST = 16;

// Audio files live in /public/audio/<key>.mp3
const PHRASES = [
  "intro",
  "good_rep",
  "incomplete_stand",
  "lean_warning",
  "arms_warning",
  "time_up",
  "step_back",
];

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

function checkSitToStand(angle: number) {
  if (angle < KNEE_SITTING_ANGLE) return "SITTING";
  if (angle > KNEE_STANDING_ANGLE) return "STANDING";
  return "TRANSITIONING";
}

function inFrame(p: { x: number; y: number; visibility?: number }) {
  return (
    (p.visibility ?? 0) >= MIN_VISIBILITY &&
    p.x > FRAME_MARGIN &&
    p.x < 1 - FRAME_MARGIN &&
    p.y > FRAME_MARGIN &&
    p.y < 1 - FRAME_MARGIN
  );
}

// Makes Arabic voice results easier to match (ابدأ / ابدا / إبدأ ...)
function normalizeArabic(t: string) {
  return t
    .replace(/[أإآٱ]/g, "ا")
    .replace(/[\u064B-\u0652]/g, "")
    .trim();
}

// ---------- Component ----------
export default function MueenCoach() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const audioRef = useRef<Record<string, HTMLAudioElement>>({});
  const lastPlayedRef = useRef<Record<string, number>>({});
  const rafRef = useRef<number>(0);
  const readyRef = useRef(false);
  const audioUnlockedRef = useRef(false);
  const bodyOkRef = useRef(false);
  const pendingStartRef = useRef(false);
  const stateRef = useRef({
    smoothedKnee: null as number | null,
    stage: null as string | null,
    reps: 0,
    testStart: null as number | null,
    testDone: false,
    minHip: 999,
    maxLean: 0,
    maxWristRatio: 0,
    okSince: null as number | null,
    lostSince: null as number | null,
  });

  const [reps, setReps] = useState(0);
  const [label, setLabel] = useState("");
  const [timeLeft, setTimeLeft] = useState<number | null>(null);
  const [ready, setReady] = useState(false);
  const [bodyWarning, setBodyWarning] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const [cameraError, setCameraError] = useState(false);

  // Mobile browsers block audio until the first tap. This "unlocks" every clip once.
  function unlockAudio() {
    if (audioUnlockedRef.current) return;
    audioUnlockedRef.current = true;
    Object.values(audioRef.current).forEach((a) => {
      a.muted = true;
      a.play()
        .then(() => {
          a.pause();
          a.currentTime = 0;
          a.muted = false;
        })
        .catch(() => {
          a.muted = false;
        });
    });
  }

  function playPhrase(key: string, force = false, cooldown = FEEDBACK_COOLDOWN_SEC) {
    const now = performance.now() / 1000;
    const last = lastPlayedRef.current[key] || 0;
    if (!force && now - last < cooldown) return;

    // Don't talk over another clip with the step-back warning
    if (
      key === "step_back" &&
      !force &&
      Object.entries(audioRef.current).some(([k, a]) => k !== "step_back" && !a.paused && !a.ended)
    ) {
      return;
    }

    if (force) {
      Object.values(audioRef.current).forEach((a) => {
        a.pause();
        a.currentTime = 0;
      });
    }
    const audio = audioRef.current[key];
    if (!audio) return;
    audio.currentTime = 0;
    audio.play().catch(() => {});
    lastPlayedRef.current[key] = now;
  }

  function beginTest() {
    const s = stateRef.current;
    playPhrase("intro", true);
    const introAudio = audioRef.current["intro"];
    const introDuration = introAudio?.duration || 5;
    s.testStart = performance.now() / 1000 + introDuration + 1.0;
    s.testDone = false;
    s.reps = 0;
    s.stage = null;
    setReps(0);
    setTimeLeft(null);
  }

  // Called by the button and by the voice command
  function startTest() {
    if (!readyRef.current) return;
    unlockAudio();

    // Body not fully visible: ask the user to step back, then start automatically
    if (!bodyOkRef.current) {
      pendingStartRef.current = true;
      setWaiting(true);
      playPhrase("step_back", true);
      return;
    }
    pendingStartRef.current = false;
    setWaiting(false);
    beginTest();
  }

  function loop() {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    const canvas = canvasRef.current;

    if (video && landmarker && canvas && video.readyState >= 2) {
      const result = landmarker.detectForVideo(video, performance.now());
      const ctx = canvas.getContext("2d")!;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const s = stateRef.current;
      const now = performance.now() / 1000;
      const w = canvas.width, h = canvas.height;

      const personSeen = !!result.landmarks && result.landmarks.length > 0;
      let bodyOk = false;

      if (personSeen) {
        const lm = result.landmarks[0];
        const pt = (i: number) => [lm[i].x * w, lm[i].y * h];

        const leftVis =
          (lm[LEFT_HIP].visibility! + lm[LEFT_KNEE].visibility! + lm[LEFT_ANKLE].visibility!) / 3;
        const rightVis =
          (lm[RIGHT_HIP].visibility! + lm[RIGHT_KNEE].visibility! + lm[RIGHT_ANKLE].visibility!) / 3;
        const useLeft = leftVis >= rightVis;

        const hipI = useLeft ? LEFT_HIP : RIGHT_HIP;
        const kneeI = useLeft ? LEFT_KNEE : RIGHT_KNEE;
        const ankleI = useLeft ? LEFT_ANKLE : RIGHT_ANKLE;
        const shoulderI = useLeft ? LEFT_SHOULDER : RIGHT_SHOULDER;
        const wristI = useLeft ? LEFT_WRIST : RIGHT_WRIST;

        // Shoulder, hip, knee and ankle must all be visible and inside the frame
        bodyOk = [shoulderI, hipI, kneeI, ankleI].every((i) => inFrame(lm[i]));

        if (bodyOk) {
          const hip = pt(hipI), knee = pt(kneeI), ankle = pt(ankleI);
          const shoulder = pt(shoulderI), wrist = pt(wristI);

          const kneeAngle = calcAngle(hip, knee, ankle);
          const hipAngle = calcAngle(shoulder, hip, knee);
          const lean = torsoLean(shoulder, hip);
          const torsoLen = Math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1]) + 1e-6;
          const wristHipRatio = Math.hypot(wrist[0] - hip[0], wrist[1] - hip[1]) / torsoLen;

          s.smoothedKnee =
            s.smoothedKnee == null ? kneeAngle : SMOOTHING * kneeAngle + (1 - SMOOTHING) * s.smoothedKnee;

          const state = checkSitToStand(s.smoothedKnee);
          const counting = s.testStart != null && !s.testDone && now >= s.testStart;

          if (state !== "SITTING") {
            s.minHip = Math.min(s.minHip, hipAngle);
            s.maxLean = Math.max(s.maxLean, lean);
            s.maxWristRatio = Math.max(s.maxWristRatio, wristHipRatio);
          }

          if (state === "SITTING") {
            s.stage = "SITTING";
            s.minHip = 999;
            s.maxLean = 0;
            s.maxWristRatio = 0;
          } else if (state === "STANDING" && s.stage === "SITTING") {
            s.stage = "STANDING";
            if (counting) {
              s.reps += 1;
              setReps(s.reps);
              const issues: string[] = [];
              if (s.minHip < HIP_FULL_EXTENSION_ANGLE) issues.push("incomplete_stand");
              if (s.maxLean > TORSO_LEAN_MAX_DEGREES) issues.push("lean_warning");
              if (s.maxWristRatio < ARM_PUSH_DISTANCE_RATIO) issues.push("arms_warning");
              playPhrase(issues[0] || "good_rep");
            }
          }
          setLabel(state);

          ctx.fillStyle = "lime";
          [hip, knee, ankle, shoulder, wrist].forEach(([x, y]) => {
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
          });
        } else {
          // Body cut off: pause rep detection and avoid a jump when they return
          s.smoothedKnee = null;
          setLabel("");
        }
      } else {
        s.smoothedKnee = null;
        setLabel("");
      }

      // ----- Body-in-frame tracking -----
      bodyOkRef.current = bodyOk;
      if (bodyOk) {
        if (s.okSince == null) s.okSince = now;
        s.lostSince = null;
      } else {
        s.okSince = null;
        if (s.lostSince == null) s.lostSince = now;
      }
      const okStable = bodyOk && s.okSince != null && now - s.okSince >= OK_AFTER_SEC;
      const lostStable = !bodyOk && s.lostSince != null && now - s.lostSince >= LOST_AFTER_SEC;

      setBodyWarning(lostStable);

      const testActive = s.testStart != null && !s.testDone;
      // Speak only when it matters (a person is half in frame, or a test is running/waiting)
      if (lostStable && (personSeen || testActive || pendingStartRef.current)) {
        playPhrase("step_back", false, STEP_BACK_COOLDOWN_SEC);
      }

      // The user asked to start earlier; begin once the full body is visible
      if (pendingStartRef.current && okStable) {
        pendingStartRef.current = false;
        setWaiting(false);
        beginTest();
      }

      // ----- Timer -----
      if (s.testStart != null && !s.testDone) {
        const elapsed = now - s.testStart;
        const remaining = Math.max(0, TEST_DURATION - elapsed);
        setTimeLeft(remaining);
        if (elapsed >= TEST_DURATION) {
          s.testDone = true;
          playPhrase("time_up", true);
        }
      }
    }
    rafRef.current = requestAnimationFrame(loop);
  }

  // Camera + pose model setup
  useEffect(() => {
    let cancelled = false;
    let stream: MediaStream | null = null;

    PHRASES.forEach((key) => {
      audioRef.current[key] = new Audio(`/audio/${key}.mp3`);
    });

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
        });
        if (cancelled) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;

        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: CAMERA_FACING },
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
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
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Voice command: "ابدأ" / "توقف"
  useEffect(() => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn("Speech recognition not supported in this browser");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "ar-SA";
    recognition.continuous = true;
    recognition.interimResults = false;

    recognition.onresult = (event: any) => {
      if (!readyRef.current) return; // ignore commands until the camera/model are ready
      const raw = event.results[event.results.length - 1][0].transcript;
      const text = normalizeArabic(raw);
      if (text.includes("ابدا")) {
        startTest();
      } else if (text.includes("توقف")) {
        pendingStartRef.current = false;
        setWaiting(false);
        stateRef.current.testDone = true;
      }
    };

    recognition.onerror = (e: any) => {
      // Mic permission denied: stop retrying forever
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        recognition.onend = null;
      }
    };

    recognition.onend = () => {
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
      recognition.onend = null;
      recognition.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      dir="rtl"
      lang="ar"
      onPointerDown={unlockAudio}
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
              <video ref={videoRef} className="hidden" playsInline muted />
              {!ready && <img src={exercisePreview} width={1200} height={912} alt="صورة توضيحية لتمرين الجلوس والوقوف" className="absolute inset-0 h-full w-full object-cover" />}
              <canvas ref={canvasRef} className={ready ? "absolute inset-0 h-full w-full object-contain" : "absolute inset-0 h-full w-full opacity-0"} aria-label="عرض الكاميرا المباشر مع نقاط تتبّع الحركة" />
              <div className="absolute right-4 top-4 flex items-center gap-2 rounded-full bg-card/95 px-4 py-2 text-base font-bold text-foreground shadow-soft sm:right-5 sm:top-5"><Camera className="size-5 text-primary" />{ready ? "الكاميرا المباشرة" : "صورة توضيحية"}</div>
              {!ready && !cameraError && <div className="absolute bottom-5 left-5 right-5 rounded-2xl border border-surface-edge bg-card/95 px-5 py-4 shadow-soft"><p className="text-lg font-bold text-foreground">بانتظار تفعيل الكاميرا</p><p className="mt-1 text-base text-muted-foreground">اسمح للمتصفح باستخدام الكاميرا لبدء التمرين.</p></div>}
              {cameraError && <div className="absolute bottom-5 left-5 right-5 rounded-2xl border border-amber-edge bg-warmsoft px-5 py-4 shadow-soft"><p className="text-lg font-bold text-amber-deep">تعذّر تشغيل الكاميرا</p><p className="mt-1 text-base text-amber-deep">تأكد من السماح باستخدام الكاميرا ثم أعد تحميل الصفحة.</p></div>}
              {ready && bodyWarning && (
                <div className="absolute bottom-5 left-5 right-5 flex items-center gap-3 rounded-2xl border border-amber-edge bg-warmsoft/95 px-5 py-4 shadow-soft" role="alert">
                  <AlertTriangle className="size-6 shrink-0 text-amber-deep" />
                  <div>
                    <p className="text-lg font-bold text-amber-deep">ابتعد قليلًا ليظهر جسمك كاملًا</p>
                    <p className="mt-0.5 text-base text-amber-deep">يجب أن تظهر الكتفان والركبتان والقدمان داخل الكاميرا.</p>
                  </div>
                </div>
              )}
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
              <div className="mt-5 flex min-h-24 items-baseline gap-4"><span className="font-num text-7xl font-semibold leading-none text-amber-deep" role="timer">{(timeLeft == null ? TEST_DURATION : Math.min(TEST_DURATION, Math.ceil(timeLeft))).toLocaleString("ar-SA")}</span><span className="text-lg font-semibold text-amber-deep">ثانية</span></div>
              <p className="mt-4 text-base font-semibold text-amber-deep">{waiting ? "بانتظار ظهور جسمك كاملًا في الكاميرا" : timeLeft === 0 ? "اكتمل التمرين، أحسنت!" : timeLeft == null ? "على مهل، ابدأ حين تكون مستعدًا" : timeLeft > TEST_DURATION ? "استعدّ، سيبدأ العدّ بعد التوجيه" : "خذ وقتك، كل حركة تُحسب"}</p>
            </div>
            <div className="mt-auto flex items-center gap-3 rounded-2xl border border-primary/25 bg-secondary px-6 py-4 text-lg font-semibold text-primary"><Mic className="size-5 shrink-0" />اضغط زر البدء، أو قل «ابدأ» ليبدأ التمرين</div>
            <Button variant="exercise" size="exercise" onClick={startTest} disabled={!ready} className="mt-3"><Play className="fill-current" />{waiting ? "بانتظار ظهور جسمك…" : timeLeft === 0 ? "ابدأ التمرين من جديد" : "ابدأ التمرين"}</Button>
          </aside>
        </main>
        <footer className="mt-9 flex flex-wrap items-center justify-between gap-3 border-t border-border/70 pt-5 text-sm text-muted-foreground"><span>معين · رفيق الحركة</span><span>بخطوات هادئة، نحو نشاط أفضل</span></footer>
      </div>
    </div>
  );
}