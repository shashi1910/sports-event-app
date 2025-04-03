import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

# Define constants
NUM_PARTICIPANTS = 300
SPORTS = ['Basketball', 'Football', 'Swimming', 'Athletics', 'Tennis', 
          'Badminton', 'Cricket', 'Chess', 'Table Tennis', 'Volleyball']
COLLEGES = ['St. Xavier\'s College', 'Christ University', 'Loyola College', 
            'Symbiosis Institute', 'BITS Pilani', 'NIT Trichy', 'IIT Madras', 
            'SRM University', 'Amity University', 'VIT Vellore',
            'Delhi University', 'Manipal Institute', 'Presidency College', 
            'Mount Carmel College', 'KIIT University', 'Sanjay Ghodawat University']
STATES = ['Karnataka', 'Tamil Nadu', 'Maharashtra', 'Delhi', 'Telangana', 
          'Kerala', 'West Bengal', 'Uttar Pradesh', 'Punjab', 'Gujarat']
START_DATE = datetime(2025, 3, 25)  # CHRISPO '25 starts on March 25, 2025
PERFORMANCE_LEVELS = ['Excellent', 'Good', 'Average', 'Below Average']

# Generate positive feedback templates
POSITIVE_FEEDBACK_TEMPLATES = [
    "The {sport} event was extremely well-organized. Loved the {adjective} atmosphere!",
    "Really enjoyed participating in {sport}. The {aspect} was particularly {adjective}.",
    "The {sport} competition was top-notch. {highlight} was the best part.",
    "Great experience at the {sport} event. The {aspect} exceeded my expectations.",
    "As a {sport} enthusiast, I found the tournament to be {adjective} and {adjective}."
]

# Generate negative feedback templates
NEGATIVE_FEEDBACK_TEMPLATES = [
    "The {sport} event needs better {aspect}. Found it rather {adjective}.",
    "Disappointed with the {aspect} during the {sport} competition. It was {adjective}.",
    "The {sport} tournament could improve on {aspect}. It was quite {adjective}.",
    "Not satisfied with the {sport} event organization. The {aspect} was {adjective}.",
    "As a {sport} player, I felt the {aspect} was below standard and {adjective}."
]

# Attributes for feedback templates
ADJECTIVES_POSITIVE = ['amazing', 'excellent', 'outstanding', 'fantastic', 'spectacular', 'wonderful', 'impressive', 'superb']
ADJECTIVES_NEGATIVE = ['disappointing', 'frustrating', 'disorganized', 'inadequate', 'poor', 'subpar', 'lackluster']
ASPECTS = ['venue', 'organization', 'officiating', 'scheduling', 'facilities', 'equipment', 'competition level', 'participant management']
HIGHLIGHTS = ['The final match', 'The semifinals', 'The awards ceremony', 'The opening event', 'The team spirit', 'The crowd support']

def generate_feedback(sport, satisfaction_level):
    """Generate realistic feedback based on satisfaction level."""
    if satisfaction_level >= 7:  # Positive feedback
        template = random.choice(POSITIVE_FEEDBACK_TEMPLATES)
        adjective = random.choice(ADJECTIVES_POSITIVE)
        aspect = random.choice(ASPECTS)
        highlight = random.choice(HIGHLIGHTS)
    else:  # Negative feedback
        template = random.choice(NEGATIVE_FEEDBACK_TEMPLATES)
        adjective = random.choice(ADJECTIVES_NEGATIVE)
        aspect = random.choice(ASPECTS)
        highlight = random.choice(HIGHLIGHTS)
    
    return template.format(sport=sport, adjective=adjective, aspect=aspect, highlight=highlight)

# Generate dataset
data = []
for i in range(NUM_PARTICIPANTS):
    participant_id = f"CHRP{i+1:03d}"
    name = fake.name()
    college = random.choice(COLLEGES)
    state = random.choice(STATES)
    sport = random.choice(SPORTS)
    day_num = random.randint(1, 5)
    event_date = (START_DATE + timedelta(days=day_num-1)).strftime('%Y-%m-%d')
    participation_time = f"{random.randint(9, 17):02d}:{random.choice(['00', '15', '30', '45']):s}"
    
    # Generate performance metrics
    satisfaction_level = random.randint(1, 10)
    performance_level = random.choice(PERFORMANCE_LEVELS)
    
    # Generate feedback
    feedback = generate_feedback(sport, satisfaction_level)
    
    data.append({
        'Participant_ID': participant_id,
        'Name': name,
        'College': college,
        'State': state,
        'Sport': sport,
        'Day': f"Day {day_num}",
        'Date': event_date,
        'Time': participation_time,
        'Performance': performance_level,
        'Feedback': feedback
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('chrispo25_data.csv', index=False)

print(f"Dataset with {len(df)} participants generated successfully!")
print(df.head())