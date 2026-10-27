// Display names, icons and descriptions for every recognised gesture (HaGRIDv2 vocabulary + motion gestures).

export const GESTURES = {
  call: { name: "Call", icon: "🤙", hint: "Thumb and little finger out" },
  dislike: { name: "Thumbs down", icon: "👎", hint: "Thumb pointing down" },
  fist: { name: "Fist", icon: "✊", hint: "All fingers closed" },
  four: { name: "Four", icon: "🖐", hint: "Four fingers up, thumb folded" },
  grabbing: { name: "Grab", icon: "🫳", hint: "Fingers curled as if grabbing" },
  grip: { name: "Grip", icon: "🤏", hint: "Fingers bent towards the thumb" },
  like: { name: "Thumbs up", icon: "👍", hint: "Thumb pointing up" },
  little_finger: { name: "Pinky", icon: "🤙", hint: "Only the little finger up" },
  middle_finger: { name: "Middle finger", icon: "🖕", hint: "Only the middle finger up" },
  mute: { name: "Mute", icon: "🤫", hint: "Index finger over the lips" },
  ok: { name: "OK", icon: "👌", hint: "Thumb and index finger form a circle" },
  one: { name: "One", icon: "☝️", hint: "Index finger up" },
  palm: { name: "Palm", icon: "🖐️", hint: "Open hand, fingers spread" },
  peace: { name: "Peace", icon: "✌️", hint: "Index and middle finger in a V" },
  peace_inverted: { name: "Peace (back)", icon: "✌️", hint: "V sign, back of the hand to the camera" },
  point: { name: "Point", icon: "👉", hint: "Index finger pointing sideways" },
  rock: { name: "Rock", icon: "🤘", hint: "Index and little finger up" },
  stop: { name: "Stop", icon: "✋", hint: "Open hand, fingers together" },
  stop_inverted: { name: "Stop (back)", icon: "✋", hint: "Stop, back of the hand to the camera" },
  three: { name: "Three", icon: "🤟", hint: "Thumb, index and middle finger up" },
  three2: { name: "Three (alt)", icon: "🖖", hint: "Index, middle and ring finger up" },
  three3: { name: "Three (alt 2)", icon: "🤟", hint: "Middle, ring and little finger up" },
  three_gun: { name: "Finger gun", icon: "🔫", hint: "Thumb up, index and middle finger forward" },
  thumb_index: { name: "L sign", icon: "🤏", hint: "Thumb and index finger at a right angle" },
  two_up: { name: "Two up", icon: "✌️", hint: "Index and middle finger up, together" },
  two_up_inverted: { name: "Two up (back)", icon: "✌️", hint: "Two up, back of the hand to the camera" },
  // Two-handed
  hand_heart: { name: "Hand heart", icon: "🫶", hint: "Both hands form a heart", twoHanded: true },
  hand_heart2: { name: "Finger heart", icon: "💞", hint: "Fingertips of both hands form a heart", twoHanded: true },
  holy: { name: "Prayer", icon: "🙏", hint: "Palms pressed together", twoHanded: true },
  take_picture: { name: "Frame", icon: "🖼️", hint: "Both hands frame a picture", twoHanded: true },
  thumb_index2: { name: "Double L", icon: "🙌", hint: "Both hands make an L", twoHanded: true },
  timeout: { name: "Timeout", icon: "⏸️", hint: "One hand on the other, forming a T", twoHanded: true },
  xsign: { name: "X sign", icon: "❌", hint: "Forearms crossed in an X", twoHanded: true },
  // Motion
  swipe_left: { name: "Swipe left", icon: "⬅️", hint: "Move an open hand quickly to the left", motion: true },
  swipe_right: { name: "Swipe right", icon: "➡️", hint: "Move an open hand quickly to the right", motion: true },
  swipe_up: { name: "Swipe up", icon: "⬆️", hint: "Move an open hand quickly upwards", motion: true },
  swipe_down: { name: "Swipe down", icon: "⬇️", hint: "Move an open hand quickly downwards", motion: true },
  pinch: { name: "Pinch", icon: "🤏", hint: "Touch thumb and index fingertips (air click)", motion: true },
  no_gesture: { name: "No gesture", icon: "·", hint: "Hand visible, no recognised gesture" },
};

export function gestureInfo(id) {
  return GESTURES[id] ?? { name: id, icon: "✦", hint: "Custom gesture", custom: true };
}
