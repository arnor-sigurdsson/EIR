import random


def simulate_health_sample(sequence_length: int) -> tuple[str, str]:
    age = random.randint(18, 80)
    fitness_score = round(random.uniform(1.0, 10.0), 1)

    cortisol_morning = random.randint(10, 25)
    cortisol_evening = random.randint(3, 12)
    sleep_quality = random.randint(1, 10)

    start_hour = random.randint(6, 10)
    start_minute = random.choice([0, 15, 30, 45])

    activity_level = random.choice(["sedentary", "light", "moderate", "vigorous"])

    input_sequence = (
        f"<START> "
        f"|Time of HR monitoring start: {start_hour:02d}:{start_minute:02d}| "
        f"|Biomarkers| "
        f"Cortisol_Morning: {cortisol_morning} - "
        f"Cortisol_Evening: {cortisol_evening} - "
        f"Sleep_Quality: {sleep_quality} - "
        f"|Demographics| "
        f"Age: {age} - "
        f"Fitness_Score: {fitness_score} - "
        f"Activity: {activity_level} -"
    )

    hr_values = []

    base_hr = 70

    if age > 60:
        base_hr += random.randint(5, 15)
    elif age < 30:
        base_hr -= random.randint(0, 10)

    if fitness_score > 7:
        base_hr -= random.randint(10, 20)
    elif fitness_score < 4:
        base_hr += random.randint(8, 18)

    activity_modifier = {
        "vigorous": random.randint(20, 40),
        "moderate": random.randint(10, 25),
        "light": random.randint(0, 10),
        "sedentary": random.randint(-5, 5),
    }[activity_level]

    base_hr += activity_modifier

    if cortisol_morning > 20:
        base_hr += random.randint(5, 12)

    if sleep_quality < 4:
        base_hr += random.randint(3, 8)

    current_hr = max(50, min(180, base_hr + random.randint(-8, 8)))

    for i in range(sequence_length):
        change = random.randint(-5, 8)

        time_of_day = (start_hour + (i * 5 / 60)) % 24

        if 6 <= time_of_day <= 9:
            change += random.randint(3, 12)
        elif 12 <= time_of_day <= 14:
            change += random.randint(2, 8)
        elif time_of_day >= 22 or time_of_day <= 5:
            change -= random.randint(5, 15)

        if activity_level == "vigorous" and 16 <= time_of_day <= 20:
            change += random.randint(8, 20)

        current_hr += change
        current_hr = max(45, min(200, current_hr))

        binned_hr = int(current_hr / 2) * 2
        hr_values.append(str(binned_hr))

    hr_sequence = " ".join(hr_values)

    return input_sequence, hr_sequence
