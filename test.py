import requests
import os

API_URL = "http://127.0.0.1:5635/tts"  # Change if needed
OUTPUT_DIR = "Benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DESCRIPTION = (
"Sunita speaks with a high pitch in a close environment. Her voice is clear, with slight dynamic changes, and the recording is of excellent quality."
#    "Mary with an expressive voice speaks with an Indian accent at a moderate pace "
#    "in a confined space with very clear audio."
)

# Sentences with increasing word counts (15 → 50 in steps of 5)
#sentences = [
#    "In a quiet meadow, sunlight streamed through the trees, and gentle winds carried the scent of blooming flowers nearby.",
#    "The small village lay nestled between rolling hills, where families gathered daily, sharing stories and laughter beside crackling fires.",
#    "On the shores of the vast, shimmering lake, children played joyfully, their shouts echoing across the still, glassy waters under a bright sky.",
#    "Under the silver moonlight, a traveler crossed the silent forest path, guided by distant lights and the whispering rustle of ancient leaves.",
#    "From the mountaintop, the world stretched endlessly, valleys and rivers weaving through lands where people lived in harmony and quiet joy.",
#    "Through bustling streets filled with colors and aromas, merchants called out cheerfully, selling spices, fabrics, and treasures from distant lands far away.",
#    "As the storm approached, clouds darkened, waves crashed violently, and the lighthouse keeper watched, ready to guide ships safely through the raging night.",
#    "In the grand library, shelves towered high with ancient books, each holding centuries of wisdom, waiting for curious minds to unlock their secrets."
#]

#sentences = [
#    # 15 words
#    "The sun dipped below the hills as birds sang softly in the gentle evening breeze.",
#    
#    # 20 words
#    "In the village square, lanterns flickered warmly as children laughed, running between stalls selling bread, fruit, and sweet cakes.",
#    
#    # 25 words
#    "Under a sky painted in shades of crimson, travelers shared stories around the fire, their voices weaving tales of bravery, love, and forgotten kingdoms.",
#    
#    # 30 words
#    "By the river’s edge, fishermen cast their nets while elders told ancient myths, each word carried by the wind across rippling waters reflecting the fading glow of the setting sun.",
#    
#    # 35 words
#    "Deep in the forest, moonlight spilled through twisted branches, revealing a hidden glade where deer grazed quietly, and fireflies drifted lazily, their soft glow shimmering like fallen stars upon the cool, damp grass.",
#    
#    # 40 words
#    "In the bustling marketplace, merchants shouted lively greetings while displaying spices, silks, and trinkets from distant lands, their voices blending with music, laughter, and the rhythmic beat of drums echoing through narrow streets lined with colorful banners and flickering lamps.",
#    
#    # 45 words
#    "High on the cliffside, the old lighthouse stood vigilant against the roaring ocean, its beam sweeping across dark waters where ships drifted cautiously, guided homeward through wind and spray toward the safe harbor, embraced by warm lights and welcoming arms ashore.",
#    
#    # 50 words
#    "Beyond the rolling plains, a caravan wound its way slowly toward the ancient city, where towering gates opened to reveal streets alive with song, color, and the mingling scents of fresh bread, fragrant flowers, and spices carried on the breeze from faraway, mysterious lands."
#]

sentences = [
    # 15 words
    "ಸಂಜೆಯ ಹೊತ್ತಿಗೆ ಬೆಟ್ಟಗಳ ಹಿಂದೆ ಸೂರ್ಯ ಅಸ್ತಮಿಸಿದಾಗ, ಹಕ್ಕಿಗಳು ಮೃದುವಾಗಿ ಹಾಡುತ್ತಾ ಗಾಳಿಯಲ್ಲಿ ತೇಲುತ್ತಿದ್ದವು.",

    # 20 words
    "ಗ್ರಾಮದ ಚೌಕದಲ್ಲಿ ದೀಪಗಳು ಹಚ್ಚಿ ಹೊಳೆಯುತ್ತಿದ್ದು, ಮಕ್ಕಳು ನಗುತಾ ಓಡಾಡಿ, ಹಣ್ಣಿನ ಹಾಗು ಸಿಹಿ ತಿಂಡಿ ಮಾರಾಟಮಾಡುತ್ತಿದ್ದರು.",

    # 25 words
    "ಕೆಂಪು ಬಣ್ಣದಲ್ಲಿ ಮುಳುಗಿದ ಆಕಾಶದ ಕೆಳಗೆ, ಪ್ರಯಾಣಿಕರು ಬೆಂಕಿಯ ಸುತ್ತ ಕೂತು, ಪ್ರೇಮ, ಧೈರ್ಯ, ಹಳೆಯ ರಾಜ್ಯಗಳ ಕಥೆಗಳನ್ನು ಹಂಚಿಕೊಂಡರು.",

    # 30 words
    "ನದಿಯ ತೀರದಲ್ಲಿ ಮೀನುಗಾರರು ಬಲೆ ಎಸೆಯುತ್ತಾ, ವಯೋವೃದ್ಧರು ಪುರಾತನ ಪೌರಾಣಿಕ ಕಥೆಗಳನ್ನು ಹೇಳುತ್ತಿದ್ದರು, ಗಾಳಿ ಆ ಮಾತುಗಳನ್ನು ಅಲೆಗಳ ಮೇಲೆ ಸಾಗಿಸುತ್ತಿತ್ತು.",

    # 35 words
    "ಆಳವಾದ ಕಾಡಿನಲ್ಲಿ ಚಂದ್ರನ ಬೆಳಕು ಕೊಂಬೆಗಳ ನಡುವೆ ಜಾರುತ್ತ, ಜಿಂಕೆಗಳು ಶಾಂತವಾಗಿ ಮೇಯುತ್ತಾ, ಮಿಂಚು ಹುಳುಗಳು ನಕ್ಷತ್ರಗಳಂತೆ ಹೊಳೆಯುತ್ತ ಹಸಿರಿನ ಮೆಟ್ಟಿಲುಗಳನ್ನು ಪ್ರಕಾಶಮಾನಗೊಳಿಸುತ್ತಿದ್ದವು.",

    # 40 words
    "ಸಂಚಾರದಿಂದ ತುಂಬಿದ ಮಾರುಕಟ್ಟೆಯಲ್ಲಿ ವ್ಯಾಪಾರಿಗಳು ಬಣ್ಣದ ವಸ್ತ್ರಗಳು, ಸುವಾಸನೆಯ ಮಸಾಲೆಗಳು ಮಾರುತ್ತಾ, ನಗುವಿನ ಧ್ವನಿಗಳು, ಸಂಗೀತ, ಡೊಳ್ಳಿನ ಶಬ್ದಗಳು ಬಣ್ಣದ ಧ್ವಜಗಳೊಂದಿಗೆ ಗಾಳಿಯಲ್ಲಿ ತೇಲುತ್ತಿದ್ದವು.",

    # 45 words
    "ಗಿರಿಶಿಖರದ ಮೇಲೆ ಹಳೆಯ ದೀಪಸ್ತಂಭ ಗರ್ಜಿಸುವ ಸಮುದ್ರದತ್ತ ಕಣ್ಣಿಡಿ ನಿಂತಿದ್ದು, ಅದರ ಬೆಳಕು ಕತ್ತಲ ಸಮುದ್ರವನ್ನು ಚೀರಿಕೊಂಡು, ಹಡಗುಗಳನ್ನು ಸುರಕ್ಷಿತ ಬಂದರಿಗೆ ಮಾರ್ಗದರ್ಶನ ಮಾಡುತ್ತಿತ್ತು.",

    # 50 words
    "ಅಪಾರ ಮೈದಾನಗಳನ್ನು ದಾಟಿ, ಒಂಟೆಗಳ ಕಾಫಿಲೆ ನಿಧಾನವಾಗಿ ಪ್ರಾಚೀನ ಪಟ್ಟಣದತ್ತ ಸಾಗುತ್ತಿತ್ತು, ಅಲ್ಲಿ ಎತ್ತರದ ಬಾಗಿಲುಗಳು ತೆರಳಿ, ಹಾಡು, ಬಣ್ಣ, ತಾಜಾ ಅನ್ನದ ಸುವಾಸನೆ, ಹೂಗಳ ಪರಿಮಳ, ಮತ್ತು ಮಸಾಲೆಗಳ ಸುಗಂಧವು ಗಾಳಿಯಲ್ಲಿ ಹರಡುತ್ತಿತ್ತು."
]

# Send sentences one by one
for idx, text in enumerate(sentences, start=1):
    print(f"[{idx}/{len(sentences)}] Sending request with {len(text.split())} words.")
    try:
        response = requests.post(API_URL, json={
            "text": text,
            "description": DESCRIPTION
        })
        
        if response.status_code == 200:
            output_path = os.path.join(OUTPUT_DIR, f"sentence_{idx}_{len(text.split())}w.wav")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Saved: {output_path}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    
    except Exception as e:
        print(f"⚠ Request failed: {e}")
