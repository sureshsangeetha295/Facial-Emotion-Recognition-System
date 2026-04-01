const CONFIG = {
  APP_NAME: 'EngageAI',
  API_BASE: '',
  PREDICT_URL: '/predict/',
  HEALTH_URL: '/health',
  DEFAULT_FPS: 5,
  MIN_FPS: 1,
  MAX_FPS: 30,
  CAPTURE_WIDTH: 640,
  CAPTURE_HEIGHT: 480,
  JPEG_QUALITY: 0.82,
  TIMELINE_MAX_POINTS: 60,
  REQUEST_TIMEOUT_MS: 30000,   
  HEALTH_INTERVAL_MS: 10000,

  EMOTION_LABELS: ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"],

  EMOTION_COLORS: {
    Anger:     "#ef4444",
    Disgust:   "#06b6d4",
    Fear:      "#8b5cf6",
    Happiness: "#f59e0b",
    Neutral:   "#6b7280",
    Sadness:   "#3b82f6",
    Surprise:  "#f97316"
  },

  // New titles for Student Engagement
  APP_TITLE: "EngageAI - Live Student Engagement Monitor",
  PAGE_TITLE: "Live Student Engagement Monitor",
  RESULT_LABEL: "CURRENT ENGAGEMENT",
  PROB_LABEL: "EMOTION PROBABILITIES",
  TIMELINE_LABEL: "SESSION TIMELINE",
  EMPTY_MESSAGE: "Start live detection to monitor student engagement",
  BOTTOM_SUBTITLE: "Live emotions will update automatically"
};