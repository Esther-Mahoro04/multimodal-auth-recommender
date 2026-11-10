"""Command-line app: face -> voice -> product prediction (Member 4).

Run from the repo root:  python scripts/main.py
"""
from config import USER_TO_CUSTOMER_ID
from face_auth import authenticate_face
from product_model import predict_product
from voice_auth import authenticate_voice


def main():
    print("=== Face + Voice Authenticated Product Recommender ===")

    # Step 1: face check
    face_path = input("Path to face image: ").strip()
    user = authenticate_face(face_path)
    if user is None:
        print("ACCESS DENIED: face not recognised.")
        return
    print(f"Face recognised: {user}")

    # Step 2: voice check (must be the same person)
    voice_path = input("Path to voice sample (e.g. 'Yes, approve'): ").strip()
    voice_user = authenticate_voice(voice_path)
    if voice_user != user:
        print("ACCESS DENIED: voice not approved.")
        return
    print("Voice verified.")

    # Step 3: product prediction
    product = predict_product(USER_TO_CUSTOMER_ID[user])
    print(f"Transaction approved. Recommended product for {user}: {product}")


if __name__ == "__main__":
    main()
