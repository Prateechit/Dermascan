"""
Dermatologist recommendation.

Builds a Google Maps search link for nearby dermatologists/skin clinics and
also serves a small fallback list of well-known hospitals in case Maps is not
available. If the browser provides the user's coordinates the link is centred
on their location; otherwise it falls back to a plain text search.
"""

# Small predefined fallback list (used when no internet / Maps unavailable).
FALLBACK_CLINICS = [
    {"name": "AIIMS Dermatology OPD, New Delhi", "phone": "011-2658-8500"},
    {"name": "Apollo Hospitals \u2013 Skin & Cosmetology", "phone": "1860-500-1066"},
    {"name": "Fortis Skin Institute", "phone": "1800-102-4444"},
    {"name": "Max Super Speciality Hospital \u2013 Dermatology", "phone": "011-2651-5050"},
]


def maps_link(lat=None, lng=None):
    """Return a Google Maps search URL for nearby dermatologists."""
    query = "dermatologist+skin+clinic+near+me"
    if lat is not None and lng is not None:
        return (
            f"https://www.google.com/maps/search/{query}/"
            f"@{lat},{lng},14z"
        )
    return f"https://www.google.com/maps/search/{query}"


def recommend(severity, lat=None, lng=None):
    """Return a recommendation payload for the frontend."""
    urgent = severity in ("critical", "high")
    return {
        "urgent": urgent,
        "message": (
            "Please consult a dermatologist soon."
            if urgent
            else "Consider a dermatologist visit if symptoms persist."
        ),
        "maps_url": maps_link(lat, lng),
        "fallback_clinics": FALLBACK_CLINICS,
    }
