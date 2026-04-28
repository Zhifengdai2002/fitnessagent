"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  AppState,
  DailyFeedbackPayload,
  Exercise,
  FitnessPlan,
  GeneratePlanPayload,
  VideoResource,
  WorkoutSession,
  generatePlan,
  getState,
  makeTomorrowPlan,
  resetApp,
  sendCoachMessage
} from "@/lib/api";

const weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

const defaultProfile: GeneratePlanPayload = {
  age: 30,
  sex: "male",
  height_cm: 175,
  weight_kg: 78,
  body_fat_pct: 24,
  fitness_level: "beginner",
  activity_level: "lightly_active",
  primary_goal: "fat_loss",
  timeline_weeks: 4,
  target_weight_kg: 72,
  target_body_fat_pct: 18,
  sessions_per_week: 4,
  minutes_per_session: 60,
  available_days: ["Monday", "Wednesday", "Saturday"],
  start_date: new Date().toISOString().slice(0, 10),
  allergies_text: "",
  dietary_preferences: [],
  profile_notes: ""
};

const defaultFeedback: DailyFeedbackPayload = {
  current_weight_kg: 78,
  current_body_fat_pct: 24,
  workout_feeling: "",
  feeling_emoji: "😊"
};

export default function Home() {
  const [state, setState] = useState<AppState>({});
  const [profile, setProfile] = useState<GeneratePlanPayload>(defaultProfile);
  const [feedback, setFeedback] = useState<DailyFeedbackPayload>(defaultFeedback);
  const [coachInput, setCoachInput] = useState("");
  const [coachOpen, setCoachOpen] = useState(true);
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    void refreshState();
  }, []);

  useEffect(() => {
    const profileInputs = state.profile_inputs;
    if (profileInputs) {
      setFeedback((current) => ({
        ...current,
        current_weight_kg: Number(profileInputs.weight_kg ?? current.current_weight_kg),
        current_body_fat_pct: Number(profileInputs.body_fat_pct ?? current.current_body_fat_pct)
      }));
    }
  }, [state.profile_inputs]);

  const currentPlan = state.agent_result?.current_plan;
  const videoResources = state.agent_result?.youtube_resources ?? [];
  const todaySession = useMemo(() => {
    const activeDate = state.active_date ?? state.agent_result?.current_date;
    return currentPlan?.workout_sessions?.find((session) => session.scheduled_date === activeDate);
  }, [currentPlan?.workout_sessions, state.active_date, state.agent_result?.current_date]);

  async function refreshState() {
    setError("");
    try {
      const response = await getState();
      setState(response.state ?? {});
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  async function handleGeneratePlan(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading("generate");
    setError("");
    try {
      const response = await generatePlan(profile);
      setState(response.state);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading("");
    }
  }

  async function handleCoachSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = coachInput.trim();
    if (!message) return;

    setLoading("chat");
    setError("");
    setCoachInput("");
    try {
      const response = await sendCoachMessage(message);
      setState(response.state);
    } catch (err) {
      setCoachInput(message);
      setError(errorMessage(err));
    } finally {
      setLoading("");
    }
  }

  async function handleMakeTomorrow(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading("tomorrow");
    setError("");
    try {
      const response = await makeTomorrowPlan(feedback);
      setState(response.state);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading("");
    }
  }

  async function handleReset() {
    setLoading("reset");
    setError("");
    try {
      const response = await resetApp();
      setState(response.state);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading("");
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div>
          <p className="eyebrow">AI Fitness Workspace</p>
          <h1>FitnessAgent</h1>
          <p className="muted">Multi-agent planning, memory, nutrition, and daily feedback.</p>
        </div>

        <form className="panel profile-form" onSubmit={handleGeneratePlan}>
          <div className="panel-title-row">
            <h2>Profile</h2>
            <button className="ghost-button" type="button" onClick={handleReset} disabled={loading === "reset"}>
              Reset
            </button>
          </div>

          <div className="field-grid">
            <label>
              Age
              <input
                type="number"
                value={profile.age}
                onChange={(event) => setProfile({ ...profile, age: Number(event.target.value) })}
              />
            </label>
            <label>
              Sex
              <select
                value={profile.sex}
                onChange={(event) => setProfile({ ...profile, sex: event.target.value as GeneratePlanPayload["sex"] })}
              >
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
                <option value="prefer_not_to_say">Prefer not to say</option>
              </select>
            </label>
            <label>
              Height
              <input
                type="number"
                value={profile.height_cm}
                onChange={(event) => setProfile({ ...profile, height_cm: Number(event.target.value) })}
              />
            </label>
            <label>
              Weight
              <input
                type="number"
                value={profile.weight_kg}
                onChange={(event) => setProfile({ ...profile, weight_kg: Number(event.target.value) })}
              />
            </label>
            <label>
              Body Fat
              <input
                type="number"
                value={profile.body_fat_pct}
                onChange={(event) => setProfile({ ...profile, body_fat_pct: Number(event.target.value) })}
              />
            </label>
            <label>
              Level
              <select
                value={profile.fitness_level}
                onChange={(event) =>
                  setProfile({ ...profile, fitness_level: event.target.value as GeneratePlanPayload["fitness_level"] })
                }
              >
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
              </select>
            </label>
          </div>

          <label>
            Available Days
            <div className="chip-grid">
              {weekdays.map((day) => {
                const checked = profile.available_days.includes(day);
                return (
                  <button
                    className={checked ? "chip active" : "chip"}
                    key={day}
                    type="button"
                    onClick={() =>
                      setProfile({
                        ...profile,
                        available_days: checked
                          ? profile.available_days.filter((item) => item !== day)
                          : [...profile.available_days, day]
                      })
                    }
                  >
                    {day.slice(0, 3)}
                  </button>
                );
              })}
            </div>
          </label>

          <div className="field-grid">
            <label>
              Sessions
              <input
                type="number"
                min={1}
                max={7}
                value={profile.sessions_per_week}
                onChange={(event) => setProfile({ ...profile, sessions_per_week: Number(event.target.value) })}
              />
            </label>
            <label>
              Start Date
              <input
                type="date"
                value={profile.start_date}
                onChange={(event) => setProfile({ ...profile, start_date: event.target.value })}
              />
            </label>
          </div>

          <label>
            Food Allergies
            <input
              value={profile.allergies_text}
              onChange={(event) => setProfile({ ...profile, allergies_text: event.target.value })}
              placeholder="peanuts, dairy..."
            />
          </label>

          <label>
            Notes
            <textarea
              value={profile.profile_notes}
              onChange={(event) => setProfile({ ...profile, profile_notes: event.target.value })}
              placeholder="Anything else I should know?"
            />
          </label>

          <button className="primary-button" disabled={loading === "generate"}>
            {loading === "generate" ? "Generating..." : "Run FitnessAgent"}
          </button>
        </form>
      </aside>

      <section className="workspace">
        {error ? <div className="error-banner">{error}</div> : null}
        {state.last_action_message ? <div className="success-banner">{state.last_action_message}</div> : null}

        <section className="hero-panel">
          <div>
            <p className="eyebrow">Today</p>
            <h2>Today&apos;s Plan</h2>
            <p className="muted">Feel free to ask AI Coach to adjust plan 😊</p>
          </div>
          <div className="date-pill">{state.active_date ?? "No plan yet"}</div>
        </section>

        <div className="main-grid">
          <section className="panel">
            <h2>{todaySession ? sessionTitle(todaySession) : "No scheduled workout"}</h2>
            {todaySession ? (
              <WorkoutCard session={todaySession} videos={videoResources} />
            ) : (
              <p className="recovery-card">Create a plan to see today&apos;s workout.</p>
            )}
          </section>

          <section className="panel">
            <h2>Today&apos;s Nutrition</h2>
            <Nutrition plan={currentPlan} />
          </section>
        </div>

        <div className="main-grid">
          <section className="panel">
            <h2>Training Cycle</h2>
            <p className="muted">{currentPlan?.summary ?? "No cycle generated yet."}</p>
            <div className="accordion-list">
              {currentPlan?.workout_sessions?.map((session) => (
                <details key={`${session.scheduled_date}-${session.focus}`} className="accordion-item">
                  <summary>{sessionTitle(session)}</summary>
                  <WorkoutCard session={session} compact videos={videoResources} />
                </details>
              ))}
            </div>
          </section>

          <form className="panel feedback-panel" onSubmit={handleMakeTomorrow}>
            <h2>Daily Feedback</h2>
            <label>
              Current Weight
              <input
                type="number"
                step="0.1"
                value={feedback.current_weight_kg}
                onChange={(event) => setFeedback({ ...feedback, current_weight_kg: Number(event.target.value) })}
              />
            </label>
            <label>
              Current Body Fat
              <input
                type="number"
                step="0.1"
                value={feedback.current_body_fat_pct}
                onChange={(event) => setFeedback({ ...feedback, current_body_fat_pct: Number(event.target.value) })}
              />
            </label>
            <label>
              How&apos;s it going?
              <textarea
                value={feedback.workout_feeling}
                onChange={(event) => setFeedback({ ...feedback, workout_feeling: event.target.value })}
                placeholder="Example: training felt okay, meals were solid, energy was a little low."
              />
            </label>
            <div className="emoji-row">
              {(["😊", "😐", "😫"] as const).map((emoji) => (
                <button
                  className={feedback.feeling_emoji === emoji ? "emoji active" : "emoji"}
                  key={emoji}
                  type="button"
                  onClick={() => setFeedback({ ...feedback, feeling_emoji: emoji })}
                >
                  {emoji}
                </button>
              ))}
            </div>
            <button className="primary-button" disabled={loading === "tomorrow"}>
              {loading === "tomorrow" ? "Saving..." : "Make Tomorrow's Plan"}
            </button>
          </form>
        </div>

        <section className="panel">
          <h2>History</h2>
          <History state={state} />
        </section>
      </section>

      <section className={coachOpen ? "coach-panel open" : "coach-panel"}>
        {coachOpen ? (
          <>
            <div className="coach-header">
              <h2>AI Coach</h2>
              <button className="ghost-button" type="button" onClick={() => setCoachOpen(false)}>
                Minimize
              </button>
            </div>
            <div className="coach-messages">
              {(state.assistant_chat_messages ?? []).map((message, index) => (
                <p key={`${message.role}-${index}`}>
                  <strong>{message.role === "user" ? "You" : "Coach"}:</strong> {message.content}
                </p>
              ))}
            </div>
            <form onSubmit={handleCoachSubmit}>
              <textarea
                value={coachInput}
                onChange={(event) => setCoachInput(event.target.value)}
                placeholder="Ask your fitness assistant..."
              />
              <div className="button-row">
                <button className="primary-button" disabled={loading === "chat"}>
                  {loading === "chat" ? "Sending..." : "Send"}
                </button>
                <button className="ghost-button" type="button" onClick={() => setCoachInput("")}>
                  Clear
                </button>
              </div>
            </form>
          </>
        ) : (
          <button className="bot-button" type="button" onClick={() => setCoachOpen(true)} aria-label="Open AI Coach">
            🤖
          </button>
        )}
      </section>
    </main>
  );
}

function WorkoutCard({
  session,
  compact = false,
  videos = []
}: {
  session: WorkoutSession;
  compact?: boolean;
  videos?: VideoResource[];
}) {
  if (session.is_cancelled) {
    return <p className="recovery-card">Workout cancelled. {session.safety_notes?.join(" ")}</p>;
  }

  const exerciseNames = new Set((session.exercises ?? []).map((exercise) => normalizeName(exercise.name ?? "")));
  const sessionVideos = videos.filter((video) => exerciseNames.has(normalizeName(video.exercise_name ?? "")));

  return (
    <div className={compact ? "workout-card compact" : "workout-card"}>
      {session.duration_minutes ? <p>Training time: {session.duration_minutes} minutes</p> : null}
      {session.warmup?.length ? <p>Warm-up: {session.warmup.join(", ")}</p> : null}
      <ul>
        {session.exercises?.map((exercise) => (
          <li key={exercise.name}>
            <strong>{exercise.name}</strong>: {exercise.sets ?? 4} x {exercise.reps ?? "10-15"}
            {!compact ? <ExerciseDetails exercise={exercise} /> : null}
          </li>
        ))}
      </ul>
      {session.cooldown?.length ? <p>Cooldown: {session.cooldown.join(", ")}</p> : null}
      {session.safety_notes?.length ? <p>Safety Notes: {session.safety_notes.join(" ")}</p> : null}
      {sessionVideos.length ? (
        <div className="video-list">
          <h3>Video Resources</h3>
          <ul>
            {sessionVideos.map((video) => (
              <li key={`${video.exercise_name}-${video.url}`}>
                <a href={video.url} target="_blank" rel="noreferrer">
                  {video.exercise_name}: {video.title ?? "demo"}
                </a>
                {video.source ? <span>{video.source}</span> : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function ExerciseDetails({ exercise }: { exercise: Exercise }) {
  const muscles = exercise.primary_muscles?.length
    ? exercise.primary_muscles.join(", ")
    : exercise.target_muscle;
  const details = [
    muscles ? { label: "Target", value: muscles } : null,
    exercise.why_this_exercise ? { label: "Why", value: exercise.why_this_exercise } : null,
    exercise.coaching_cue ? { label: "Cue", value: exercise.coaching_cue } : null,
    exercise.common_mistake ? { label: "Watch", value: exercise.common_mistake } : null,
    exercise.regression ? { label: "Regression", value: exercise.regression } : null,
    exercise.progression ? { label: "Progression", value: exercise.progression } : null
  ].filter(Boolean) as { label: string; value: string }[];

  if (!details.length) return null;

  return (
    <dl className="exercise-details">
      {details.map((item) => (
        <div key={item.label}>
          <dt>{item.label}</dt>
          <dd>{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function normalizeName(value: string) {
  return value.trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
}

function Nutrition({ plan }: { plan?: FitnessPlan }) {
  const targets = plan?.nutrition_targets ?? {};
  const meals = plan?.meal_suggestions ?? [];
  return (
    <div>
      {Object.keys(targets).length ? (
        <p>
          Daily calories: {targets.daily_calories ?? targets.calories}, protein: {targets.protein_g}g, carbs:{" "}
          {targets.carbs_g}g, fat: {targets.fat_g}g
        </p>
      ) : (
        <p className="muted">Nutrition targets will appear after plan generation.</p>
      )}
      <ul>
        {meals.map((meal: { meal?: string; food_name?: string; serving_size?: string }, index: number) => (
          <li key={`${meal.meal}-${index}`}>
            <strong>{meal.meal}</strong>: {meal.food_name} {meal.serving_size ? `(${meal.serving_size})` : ""}
          </li>
        ))}
      </ul>
    </div>
  );
}

function History({ state }: { state: AppState }) {
  const history = state.daily_history ?? state.agent_result?.daily_history ?? [];
  if (!history.length) return <p className="muted">No history yet.</p>;

  const grouped = history.reduce<Record<string, typeof history>>((acc, item) => {
    const key = `Cycle ${item.cycle_number ?? 1}`;
    acc[key] = [...(acc[key] ?? []), item];
    return acc;
  }, {});

  return (
    <div className="accordion-list">
      {Object.entries(grouped).map(([cycle, items]) => (
        <details className="accordion-item" key={cycle} open>
          <summary>{cycle}</summary>
          <ul>
            {items.map((item) => (
              <li key={item.date}>
                <strong>{item.date}</strong> {item.feedback?.emoji} {item.weight_kg} kg, {item.body_fat_pct}% body fat ·{" "}
                {item.status === "cancelled" ? "Workout cancelled" : item.completed_actions?.join(", ") || "No scheduled workout"}
              </li>
            ))}
          </ul>
        </details>
      ))}
    </div>
  );
}

function sessionTitle(session: WorkoutSession) {
  return `${session.day ?? ""} ${session.scheduled_date ?? ""} · ${session.focus ?? "Workout"}`.trim();
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "Something went wrong.";
}
