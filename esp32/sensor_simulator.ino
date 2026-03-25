/*
 * ============================================================================
 * EdgeHAR — ESP32 IMU Sensor Simulator
 * ============================================================================
 *
 * Simulates 6-channel IMU data (3-axis accelerometer + 3-axis gyroscope)
 * and sends it as JSON over Serial at 125Hz (every 8ms).
 *
 * Output Format (JSON):
 *   {"ax":0.12,"ay":0.98,"az":0.05,"gx":0.01,"gy":-0.02,"gz":0.00,"ts":12345}
 *
 * Sensor Channels:
 *   ax, ay, az — Accelerometer X, Y, Z (in g, ±2g range)
 *   gx, gy, gz — Gyroscope X, Y, Z (in rad/s, ±250°/s range)
 *   ts         — Timestamp in milliseconds since boot
 *
 * Activity Patterns (cycles every 5 seconds):
 *   0: WALKING          — Regular sinusoidal gait pattern
 *   1: WALKING_UPSTAIRS — Higher frequency, increased vertical accel
 *   2: WALKING_DOWNSTAIRS — Lower frequency, negative vertical component
 *   3: SITTING          — Near-zero motion, slight drift
 *   4: STANDING         — Minimal motion, gravity on Z-axis
 *   5: LAYING           — Gravity shifted to X-axis, no motion
 *
 * Hardware: ESP32 (or any Arduino-compatible board)
 * Baud Rate: 115200
 * ============================================================================
 */

// ─── Configuration ──────────────────────────────────────────────────────────
#define SERIAL_BAUD     115200
#define SAMPLE_INTERVAL 8       // ms between samples (125Hz)
#define ACTIVITY_DURATION 5000  // ms per activity (5 seconds)
#define NUM_ACTIVITIES  6

// ─── Global Variables ───────────────────────────────────────────────────────
unsigned long lastSampleTime = 0;
unsigned long startTime = 0;
int currentActivity = 0;

// Activity names for debugging
const char* activityNames[] = {
  "WALKING",
  "WALKING_UPSTAIRS",
  "WALKING_DOWNSTAIRS",
  "SITTING",
  "STANDING",
  "LAYING"
};

// ─── Setup ──────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) {
    ; // Wait for serial port to connect (needed for USB-native boards)
  }

  startTime = millis();

  Serial.println("// ========================================");
  Serial.println("// EdgeHAR ESP32 Sensor Simulator");
  Serial.println("// 6-channel IMU @ 125Hz");
  Serial.println("// ========================================");
  Serial.print("// Starting with activity: ");
  Serial.println(activityNames[currentActivity]);
}

// ─── Main Loop ──────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();

  // Maintain 125Hz sampling rate
  if (now - lastSampleTime < SAMPLE_INTERVAL) {
    return;
  }
  lastSampleTime = now;

  // Cycle through activities every ACTIVITY_DURATION ms
  unsigned long elapsed = now - startTime;
  int newActivity = (elapsed / ACTIVITY_DURATION) % NUM_ACTIVITIES;

  if (newActivity != currentActivity) {
    currentActivity = newActivity;
    // Print activity change as a comment (won't break JSON parsing)
    Serial.print("// Activity changed to: ");
    Serial.println(activityNames[currentActivity]);
  }

  // Generate simulated sensor data based on current activity
  float ax, ay, az, gx, gy, gz;
  generateSensorData(currentActivity, now, &ax, &ay, &az, &gx, &gy, &gz);

  // Send JSON over serial
  sendJSON(ax, ay, az, gx, gy, gz, now);
}

// ─── Sensor Data Generation ─────────────────────────────────────────────────

/**
 * Generate synthetic IMU data for a given activity.
 *
 * Each activity has a distinct pattern based on real-world biomechanics:
 *   - Walking: ~2Hz sinusoidal pattern on all axes, moderate amplitude
 *   - Upstairs: Higher effort (larger accel), slightly faster
 *   - Downstairs: Impact-heavy (spiky accel), moderate frequency
 *   - Sitting: Nearly static, small random drift
 *   - Standing: Static with gravity on Z, tiny sway
 *   - Laying: Gravity on X-axis, minimal movement
 *
 * @param activity  Current activity index (0-5)
 * @param timeMs    Current time in milliseconds
 * @param ax-gz     Output pointers for 6 sensor channels
 */
void generateSensorData(int activity, unsigned long timeMs,
                         float* ax, float* ay, float* az,
                         float* gx, float* gy, float* gz) {
  // Convert time to seconds for sine calculations
  float t = timeMs / 1000.0;

  // Small noise component
  float noise = ((float)random(-100, 100)) / 10000.0;

  switch (activity) {

    case 0: // WALKING
      // Regular gait pattern ~2Hz
      // Vertical (az) has double frequency from left-right steps
      *ax = 0.15 * sin(2.0 * PI * 2.0 * t) + noise;          // Forward/backward sway
      *ay = 0.10 * sin(2.0 * PI * 2.0 * t + PI/3) + noise;   // Lateral sway
      *az = 0.98 + 0.20 * sin(2.0 * PI * 4.0 * t) + noise;   // Vertical bounce (gravity + gait)
      *gx = 0.05 * sin(2.0 * PI * 2.0 * t) + noise;          // Roll from hip movement
      *gy = 0.08 * cos(2.0 * PI * 2.0 * t) + noise;          // Pitch from leg swing
      *gz = 0.03 * sin(2.0 * PI * 1.0 * t) + noise;          // Yaw from arm swing
      break;

    case 1: // WALKING UPSTAIRS
      // Higher effort, increased vertical acceleration
      *ax = 0.20 * sin(2.0 * PI * 2.5 * t) + noise;          // More forward lean
      *ay = 0.12 * sin(2.0 * PI * 2.5 * t + PI/4) + noise;   // Lateral
      *az = 0.95 + 0.35 * sin(2.0 * PI * 5.0 * t) + noise;   // Stronger vertical push
      *gx = 0.10 * sin(2.0 * PI * 2.5 * t) + noise;          // More roll
      *gy = 0.15 * cos(2.0 * PI * 2.5 * t) + noise;          // Significant pitch (climbing)
      *gz = 0.04 * sin(2.0 * PI * 1.25 * t) + noise;         // Yaw
      break;

    case 2: // WALKING DOWNSTAIRS
      // Impact-heavy, negative vertical component
      *ax = 0.18 * sin(2.0 * PI * 1.8 * t) + noise;          // Forward lean
      *ay = 0.14 * sin(2.0 * PI * 1.8 * t + PI/6) + noise;   // Lateral balance
      *az = 1.02 - 0.30 * abs(sin(2.0 * PI * 3.6 * t)) + noise; // Impact spikes
      *gx = 0.07 * sin(2.0 * PI * 1.8 * t) + noise;          // Roll
      *gy = -0.12 * cos(2.0 * PI * 1.8 * t) + noise;         // Negative pitch (descending)
      *gz = 0.05 * sin(2.0 * PI * 0.9 * t) + noise;          // Yaw
      break;

    case 3: // SITTING
      // Nearly static, small random drift
      *ax = 0.02 * sin(2.0 * PI * 0.1 * t) + noise;          // Tiny forward lean
      *ay = 0.01 * sin(2.0 * PI * 0.15 * t) + noise;         // Minimal lateral
      *az = 0.98 + 0.01 * sin(2.0 * PI * 0.2 * t) + noise;   // Gravity + breathing
      *gx = 0.005 * sin(2.0 * PI * 0.1 * t) + noise;         // Negligible rotation
      *gy = 0.003 * cos(2.0 * PI * 0.1 * t) + noise;
      *gz = 0.002 * sin(2.0 * PI * 0.05 * t) + noise;
      break;

    case 4: // STANDING
      // Static with gravity on Z-axis, tiny sway
      *ax = 0.03 * sin(2.0 * PI * 0.3 * t) + noise;          // Postural sway
      *ay = 0.02 * sin(2.0 * PI * 0.25 * t + PI/2) + noise;  // Lateral sway
      *az = 0.99 + 0.015 * sin(2.0 * PI * 0.2 * t) + noise;  // Gravity dominant
      *gx = 0.008 * sin(2.0 * PI * 0.3 * t) + noise;         // Small roll correction
      *gy = 0.006 * cos(2.0 * PI * 0.3 * t) + noise;         // Small pitch correction
      *gz = 0.003 * sin(2.0 * PI * 0.15 * t) + noise;        // Minimal yaw
      break;

    case 5: // LAYING
      // Gravity shifted to X-axis, no motion
      *ax = 0.97 + 0.005 * sin(2.0 * PI * 0.1 * t) + noise;  // Gravity on X (laying on back)
      *ay = 0.01 * sin(2.0 * PI * 0.08 * t) + noise;          // Minimal lateral
      *az = 0.05 + 0.005 * sin(2.0 * PI * 0.12 * t) + noise;  // Near-zero vertical
      *gx = 0.002 * sin(2.0 * PI * 0.05 * t) + noise;         // Negligible rotation
      *gy = 0.001 * cos(2.0 * PI * 0.05 * t) + noise;
      *gz = 0.001 * sin(2.0 * PI * 0.03 * t) + noise;
      break;
  }
}

// ─── JSON Output ────────────────────────────────────────────────────────────

/**
 * Send sensor data as JSON string over Serial.
 *
 * Format: {"ax":0.12,"ay":0.98,"az":0.05,"gx":0.01,"gy":-0.02,"gz":0.00,"ts":12345}
 *
 * @param ax  Accelerometer X (g)
 * @param ay  Accelerometer Y (g)
 * @param az  Accelerometer Z (g)
 * @param gx  Gyroscope X (rad/s)
 * @param gy  Gyroscope Y (rad/s)
 * @param gz  Gyroscope Z (rad/s)
 * @param ts  Timestamp (ms since boot)
 */
void sendJSON(float ax, float ay, float az,
              float gx, float gy, float gz,
              unsigned long ts) {
  // Build JSON string manually for speed (no ArduinoJson dependency)
  Serial.print("{\"ax\":");
  Serial.print(ax, 4);
  Serial.print(",\"ay\":");
  Serial.print(ay, 4);
  Serial.print(",\"az\":");
  Serial.print(az, 4);
  Serial.print(",\"gx\":");
  Serial.print(gx, 4);
  Serial.print(",\"gy\":");
  Serial.print(gy, 4);
  Serial.print(",\"gz\":");
  Serial.print(gz, 4);
  Serial.print(",\"ts\":");
  Serial.print(ts);
  Serial.println("}");
}
