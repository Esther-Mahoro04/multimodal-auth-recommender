# Unauthorized face samples

Images here are used by `python scripts/main.py --demo` as unauthorized face attempts.
They live in this subfolder so `image_processing.py` never enrolls them as team members.

- `no_face_object.jpg` – a generated picture with no face in it (tests the "no face detected" rejection).
- Add a photo of a person outside the team (with their permission), e.g. `stranger1.jpg`,
  to test the "unknown face" rejection. The demo runs every image in this folder.
