import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os
import math
import pygame
import base64
import requests
import json
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as compare_ssim

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize Pygame for particle effects
pygame.init()
particle_screen = pygame.Surface((1000, 800), pygame.SRCALPHA)

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyDwm1AeQx-a2w5PP8BpW9oIeKRLWlUFmN8"  # Replace with your actual API key
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key={GEMINI_API_KEY}"

# Malayalam letters pool
MALAYALAM_LETTERS = [
    "അ", "ആ", "ഇ",  "ഉ",  "എ", "ഏ", "ഒ","ക്ഷ" 
    "ക", "ഖ", "ഗ", "ഘ", "ങ", "ച", "ഛ", "ജ", "ഝ", "ഞ",
    "ട", "ഠ", "ഡ", "ഢ", "ണ", "ത", "ഥ", "ദ", "ധ", "ന", "പ"
]

# Font setup for Malayalam
def get_malayalam_font(size=100):
    """Try to find a Malayalam font on the system"""
    font_paths = [
        "D:/fonts/Noto_Sans_Malayalam/NotoSansMalayalam-VariableFont_wdth,wght.ttf",
        "D:/fonts/Noto_Sans_Malayalam/NotoSansMalayalam-Regular.ttf"
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                pass
    
    return ImageFont.load_default()

# Create reference letter images
def create_reference_letter(letter, size=200):
    """Create a reference image of the Malayalam letter"""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Get Malayalam font
    font = get_malayalam_font(int(size * 0.8))
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the letter
    position = ((size - text_width) // 2, (size - text_height) // 2)
    
    # Draw the letter in white
    draw.text(position, letter, fill=(255, 255, 255), font=font)
    
    # Convert to OpenCV format
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Create particles for visual effects
def create_particles(x, y, color=(255, 255, 255)):
    """Create explosion particles at given position"""
    particles = []
    for _ in range(50):
        angle = random.uniform(0, math.pi * 2)
        speed = random.uniform(1, 5)
        size = random.randint(2, 8)
        lifetime = random.randint(20, 40)
        particles.append({
            'x': x, 'y': y,
            'vx': math.cos(angle) * speed,
            'vy': math.sin(angle) * speed,
            'color': color,
            'size': size,
            'life': lifetime
        })
    return particles

# Update and draw particles
def update_particles(particles):
    """Update and draw particles"""
    particle_screen.fill((0, 0, 0, 0))
    new_particles = []
    
    for p in particles:
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['life'] -= 1
        
        if p['life'] > 0:
            alpha = min(255, p['life'] * 6)
            color = (*p['color'], alpha)
            pygame.draw.circle(
                particle_screen, 
                color, 
                (int(p['x']), int(p['y'])), 
                p['size']
            )
            new_particles.append(p)
    
    return new_particles

# Function to draw Malayalam text using PIL
def draw_malayalam_text(image, text, position, font_size, color):
    """Draw Malayalam text on an OpenCV image using PIL"""
    # Convert OpenCV image to PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Get Malayalam font
    font = get_malayalam_font(font_size)
    
    # Draw the text
    draw.text(position, text, font=font, fill=color)
    
    # Convert back to OpenCV
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Local funny comment generator
def get_funny_comment(accuracy):
    """Generate funny Malayalam comment based on accuracy"""
    if accuracy > 85:
        return random.choice([
            "അയ്യോ പടിക്കൂ! നീ ഒരു കലാകാരൻ ആണോ?",
            "ഇത് ഡ്രോയിംഗ് അല്ല, സ്കെൻ കോപ്പി!",
            "എന്റെ മോനെ, നീ ആദ്യം എന്നെ ഒന്ന് മെച്ചപ്പെടുത്തണം!",
            "അച്ചടിച്ചത് പോലെയുള്ള എഴുത്ത്!",
            "ഗുരു ദക്ഷിണ എത്ര വേണം?"
        ])
    elif accuracy > 60:
        return random.choice([
            "കുറച്ച് പരിശീലനം കൂടി വേണ്ടിയിരുന്നു!",
            
            "ഇത്രേം നന്നായാൽ മതി, ഇനി മെച്ചപ്പെടുത്താൻ പറ്റില്ല!",
            "കണ്ണാടി മുന്നിൽ നിന്ന് എഴുതിയതല്ലേ?",
            "ശരിക്കും ശ്രമിച്ചിട്ടുണ്ട്!"
        ])
    elif accuracy > 40:
        return random.choice([
            "അക്ഷരത്തിന്റെ ആത്മാവ് എവിടെ?",
            "ഒന്നുകൂടി ശ്രമിക്കൂ... എന്നെ പരിഹസിക്കാനല്ല!",
            "ഇതു കണ്ട് എഴുത്തുകാരൻ കരയും!",
            "എന്റെ കുഞ്ഞേ, ഈ അക്ഷരം എങ്ങനെയാണെന്ന് നോക്കിയിട്ട് വരയ്ക്ക്!",
            "ശരിക്കും ശ്രമിച്ചാൽ സാധിക്കും!"
        ])
    else:
        return random.choice([
            "ഇതെന്താ മോനേ? കാലിൽ കെട്ടിയ കുറ്റി ആണോ?",
            "അക്ഷരം അല്ല ഇത്, അപകടസൂചന!",
            "ഇത് കണ്ട് എന്റെ കണ്ണുകൾ കഴപ്പിച്ചു!",
            "നിങ്ങൾ ശരിക്കും മലയാളം അറിയുമോ?",
            "അധ്യാപകർക്ക് ഇതു കണ്ടാൽ രോഷം വരും!"
        ])

# Analyze drawing with Gemini API
def analyze_with_gemini(drawn_image, reference_image, letter):
    """Send images to Gemini for analysis"""
    # Convert images to base64
    _, drawn_encoded = cv2.imencode('.jpg', drawn_image)
    drawn_base64 = base64.b64encode(drawn_encoded).decode('utf-8')
    
    _, ref_encoded = cv2.imencode('.jpg', reference_image)
    ref_base64 = base64.b64encode(ref_encoded).decode('utf-8')
    
    # Create structured prompt
    prompt = f"""
    ROLE: You are a strict Malayalam handwriting expert evaluating children's letter drawings.
    TASK: Compare user's drawing against reference for letter '{letter}'.
    
    ANALYSIS CRITERIA:
    1. Structural accuracy (40%): Shape, proportions, curves
    2. Stroke accuracy (30%): Direction, connection points
    3. Feature completeness (30%): All critical elements present
    
    OUTPUT REQUIREMENTS:
    - "accuracy": 0-100 score (average of criteria scores)
    - "malayalam_comment": Funny 1-sentence feedback in Malayalam
    - "english_analysis": 2-3 sentence technical analysis
    
    RESPONSE FORMAT: Pure JSON only with EXACTLY these keys:
    {{
        "accuracy": integer,
        "malayalam_comment": string,
        "english_analysis": string
    }}
    """
    
    # Create request payload
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": ref_base64
                        }
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": drawn_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,  # Lower for more accuracy
            "topK": 1,
            "maxOutputTokens": 300
        }
    }
    
    # Send request to Gemini
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        response_data = response.json()
        
        # Extract the JSON response from Gemini
        if 'candidates' in response_data and response_data['candidates']:
            content = response_data['candidates'][0]['content']
            if 'parts' in content and content['parts']:
                text_response = content['parts'][0]['text']
                
                # Extract JSON from the response
                try:
                    # Find JSON part in the response
                    start_idx = text_response.find('{')
                    end_idx = text_response.rfind('}') + 1
                    json_str = text_response[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    # Validate response format
                    if all(key in result for key in ["accuracy", "malayalam_comment", "english_analysis"]):
                        return result
                except:
                    pass
    except Exception as e:
        print(f"Gemini API error: {e}")
    
    # Fallback: Calculate local accuracy using structural similarity
    try:
        # Resize images to same dimensions
        ref_gray = cv2.cvtColor(cv2.resize(reference_image, (200, 200)), cv2.COLOR_BGR2GRAY)
        drawn_gray = cv2.cvtColor(cv2.resize(drawn_image, (200, 200)), cv2.COLOR_BGR2GRAY)
        
        # Compute Structural Similarity Index
        ssim_score = compare_ssim(ref_gray, drawn_gray)
        accuracy = int(ssim_score * 100)
        
        # Generate funny comment based on accuracy
        malayalam_comment = get_funny_comment(accuracy)
        
        return {
            "accuracy": accuracy,
            "malayalam_comment": malayalam_comment,
            "english_analysis": f"Local fallback: SSIM score {ssim_score:.2f}"
        }
    except:
        # Final fallback if everything fails
        accuracy = random.randint(30, 70)
        return {
            "accuracy": accuracy,
            "malayalam_comment": get_funny_comment(accuracy),
            "english_analysis": "Technical error occurred in analysis"
        }

# Main application
def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    w, h = 1000, 800
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    
    # Drawing variables
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    points = []
    drawing = False
    
    # Game variables
    current_letter = random.choice(MALAYALAM_LETTERS)
    reference_img = create_reference_letter(current_letter, 200)
    particles = []
    gemini_response = None
    message_time = 0
    show_comparison = False
    loading = False
    loading_start = 0
    
    # Create a window for comparison
    cv2.namedWindow("Letter Comparison", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Letter Comparison", 600, 300)
    
    # Store reference image for display
    ref_display = reference_img.copy()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip and resize frame
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (w, h))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            nose_tip = face_landmarks.landmark[4]  # Nose tip landmark
            
            # Convert to pixel coordinates
            x = int(nose_tip.x * w)
            y = int(nose_tip.y * h)
            
            # Draw nose cursor
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 15, (0, 255, 255), 2)
            
            # Start drawing when 'd' pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('d'):
                drawing = True
                # Clear canvas when starting new drawing
                if not points:
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                
            # Stop drawing when 's' pressed
            if key == ord('s') and drawing:
                drawing = False
                show_comparison = True
                loading = True
                loading_start = time.time()
                
                # Create small version for Gemini
                drawn_small = cv2.resize(canvas, (200, 200))
                ref_small = cv2.resize(ref_display, (200, 200))
                
                # Send to Gemini in a separate thread (simplified for demo)
                gemini_response = analyze_with_gemini(drawn_small, ref_small, current_letter)
                
                # Create comparison image
                comparison_img = np.zeros((300, 600, 3), dtype=np.uint8)
                
                # Add reference letter
                ref_x, ref_y = 50, 50
                comparison_img[ref_y:ref_y+200, ref_x:ref_x+200] = ref_small
                cv2.putText(comparison_img, "Reference", (ref_x+50, ref_y+230), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add drawn letter
                drawn_x, drawn_y = 350, 50
                comparison_img[drawn_y:drawn_y+200, drawn_x:drawn_x+200] = drawn_small
                cv2.putText(comparison_img, "Your Drawing", (drawn_x+40, drawn_y+230), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Generate particles
                particles = create_particles(x, y)
                
                # Reset for next letter
                points = []
                message_time = time.time()
                loading = False
            
            # Reset when 'n' pressed
            if key == ord('n'):
                drawing = False
                show_comparison = False
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                points = []
                current_letter = random.choice(MALAYALAM_LETTERS)
                ref_display = create_reference_letter(current_letter, 200)
                gemini_response = None
            
            # Draw when enabled
            if drawing:
                points.append((x, y))
        
        # Draw the white trail
        for i in range(1, len(points)):
            cv2.line(canvas, points[i-1], points[i], (255, 255, 255), 6)
        
        # Add particle effects
        if particles:
            particles = update_particles(particles)
            particle_np = pygame.surfarray.array3d(particle_screen)
            particle_np = np.transpose(particle_np, (1, 0, 2))
            frame = cv2.addWeighted(frame, 0.7, particle_np, 0.3, 0)
        
        # Combine frame and canvas
        frame = cv2.add(frame, canvas)
        
        # Display current reference letter in top-right corner
        ref_small = cv2.resize(ref_display, (100, 100))
        frame[10:110, w-110:w-10] = ref_small
        
        # Add label for reference letter
        cv2.putText(frame, "Reference Letter", (w-200, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # UI Improvements
        # Create a semi-transparent panel for controls
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-120), (400, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Display instructions
        cv2.putText(frame, "Press 'd' to start drawing", (20, h - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press 's' to submit", (20, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'n' for new letter", (20, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display current letter to draw
        frame = draw_malayalam_text(
            frame, 
            f"ഈ അക്ഷരം വരയ്ക്കുക: {current_letter}", 
            (20, 30), 
            30, 
            (0, 255, 255)
        )
        
        # Display Gemini analysis
        if gemini_response and (time.time() - message_time < 10):
            # Display accuracy
            accuracy = gemini_response.get('accuracy', 0)
            accuracy_color = (0, 200, 0) if accuracy > 70 else (0, 100, 255) if accuracy > 40 else (0, 0, 255)
            cv2.putText(frame, f"Accuracy: {accuracy}%", 
                       (w//2 - 100, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, accuracy_color, 2)
            
            # Display Malayalam comment
            malayalam_comment = gemini_response.get('malayalam_comment', '')
            if malayalam_comment:
                frame = draw_malayalam_text(
                    frame, 
                    malayalam_comment, 
                    (w//2 - 300, 200), 
                    30, 
                    (255, 255, 0)
                )
            
            # Display English analysis
            english_analysis = gemini_response.get('english_analysis', '')
            if english_analysis:
                # Split into multiple lines if needed
                words = english_analysis.split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + word + " "
                    (width, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    if width < w - 100:
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = word + " "
                
                if current_line:
                    lines.append(current_line)
                
                # Display each line
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (w//2 - 300, 250 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
        
        # Show loading indicator
        if loading:
            elapsed = time.time() - loading_start
            dots = "." * (int(elapsed * 2) % 4)
            cv2.putText(frame, f"Analyzing with Gemini{dots}", 
                       (w//2 - 150, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Show comparison window if active
        if show_comparison and gemini_response:
            # Add analysis to comparison image
            cv2.putText(comparison_img, f"Accuracy: {gemini_response.get('accuracy', 0)}%", 
                       (230, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Add Malayalam comment
            comparison_img = draw_malayalam_text(
                comparison_img, 
                gemini_response.get('malayalam_comment', ''),
                (50, 280), 
                20, 
                (255, 255, 0)
            )
            
            cv2.imshow("Letter Comparison", comparison_img)
        else:
            # Close comparison window if not active
            if cv2.getWindowProperty("Letter Comparison", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Letter Comparison")
        
        # Show frame
        cv2.imshow("Mookku Lipi: Write Malayalam with Your Nose", frame)
        
        # Exit on 'q'
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()