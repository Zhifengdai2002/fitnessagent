export type ApiResponse = {
  ok: boolean;
  message: string;
  state: AppState;
};

export type ChatResponse = ApiResponse & {
  reply: string;
};

export type ChatMessage = {
  role: "user" | "assistant" | string;
  content: string;
};

export type Exercise = {
  name?: string;
  target_muscle?: string;
  sets?: number;
  reps?: string;
  equipment?: string;
  notes?: string;
  primary_muscles?: string[];
  secondary_muscles?: string[];
  coaching_cue?: string;
  why_this_exercise?: string;
  common_mistake?: string;
  regression?: string;
  progression?: string;
  knowledge_source?: string;
};

export type WorkoutSession = {
  scheduled_date?: string;
  day?: string;
  focus?: string;
  duration_minutes?: number;
  warmup?: string[];
  exercises?: Exercise[];
  cooldown?: string[];
  safety_notes?: string[];
  is_cancelled?: boolean;
};

export type VideoResource = {
  exercise_name?: string;
  title?: string;
  url?: string;
  source?: string;
};

export type MealSuggestion = {
  meal?: string;
  food_name?: string;
  serving_size?: string;
};

export type FitnessPlan = {
  summary?: string;
  cycle_number?: number;
  cycle_start_date?: string;
  cycle_end_date?: string;
  workout_sessions?: WorkoutSession[];
  nutrition_targets?: Record<string, number | string>;
  meal_suggestions?: MealSuggestion[];
};

export type AgentResult = {
  current_date?: string;
  current_plan?: FitnessPlan;
  daily_history?: DailyHistoryItem[];
  coaching_message?: string;
  youtube_resources?: VideoResource[];
};

export type DailyHistoryItem = {
  date?: string;
  cycle_number?: number;
  plan_focus?: string;
  status?: string;
  weight_kg?: number;
  body_fat_pct?: number;
  completed_actions?: string[];
  feedback?: {
    emoji?: string;
    emoji_label?: string;
    workout_feeling?: string;
    injury_areas?: string[];
  };
};

export type AppState = {
  active_date?: string;
  profile_inputs?: Record<string, unknown> | null;
  agent_result?: AgentResult;
  memory_store?: Record<string, unknown>;
  daily_history?: DailyHistoryItem[];
  assistant_chat_messages?: ChatMessage[];
  last_feedback_summary?: string;
  last_action_message?: string;
};

export type GeneratePlanPayload = {
  age: number;
  sex: "male" | "female" | "other" | "prefer_not_to_say";
  height_cm: number;
  weight_kg: number;
  body_fat_pct: number;
  fitness_level: "beginner" | "intermediate" | "advanced";
  activity_level: string;
  primary_goal: string;
  timeline_weeks: number;
  target_weight_kg: number;
  target_body_fat_pct: number;
  sessions_per_week: number;
  minutes_per_session: number;
  available_days: string[];
  start_date: string;
  allergies_text: string;
  dietary_preferences: string[];
  profile_notes: string;
};

export type DailyFeedbackPayload = {
  current_weight_kg: number;
  current_body_fat_pct: number;
  workout_feeling: string;
  feeling_emoji: "😊" | "😐" | "😫";
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    cache: "no-store"
  });

  if (!response.ok) {
    let detail = `Request failed with ${response.status}`;
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {
      // Keep the status-based fallback.
    }
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export function getState() {
  return request<ApiResponse>("/state");
}

export function generatePlan(payload: GeneratePlanPayload) {
  return request<ApiResponse>("/generate_plan", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function sendCoachMessage(message: string) {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({ message })
  });
}

export function makeTomorrowPlan(payload: DailyFeedbackPayload) {
  return request<ApiResponse>("/make_tomorrow_plan", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function resetApp() {
  return request<ApiResponse>("/reset", {
    method: "POST"
  });
}
