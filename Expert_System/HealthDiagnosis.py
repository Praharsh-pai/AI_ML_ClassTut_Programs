class MedicalDiagnosisExpertSystem:
    def __init__(self):
        self.rules = {
            "flu": [
                ("Do you have a fever?", "yes", 3),
                ("Are you experiencing chills?", "yes", 2),
                ("Do you have a cough?", "yes", 2),
                ("Are you feeling fatigued?", "yes", 1),
            ],
            "common cold": [
                ("Do you have a runny nose?", "yes", 2),
                ("Are you sneezing?", "yes", 2),
                ("Do you have a sore throat?", "yes", 1),
            ],
            "allergy": [
                ("Are you experiencing sneezing?", "yes", 3),
                ("Do you have itchy eyes?", "yes", 2),
                ("Have you been exposed to pollen or dust?", "yes", 2),
            ],
            "stomach flu": [
                ("Are you experiencing nausea?", "yes", 3),
                ("Do you have diarrhea?", "yes", 2),
                ("Do you have abdominal cramps?", "yes", 2),
            ],
            "none": [
                ("Do you have any symptoms?", "no", 3),
            ],
        }

    def ask_question(self, question):
        answer = input(question + " (yes/no): ").strip().lower()
        return answer == "yes"

    def diagnose(self):
        diagnosis_scores = {key: 0 for key in self.rules.keys()}
        
        for diagnosis, questions in self.rules.items():
            print(f"\nChecking for {diagnosis}...")
            for question, expected_answer, weight in questions:
                if self.ask_question(question) == (expected_answer == "yes"):
                    diagnosis_scores[diagnosis] += weight
        
        sorted_diagnoses = sorted(diagnosis_scores.items(), key=lambda x: x[1], reverse=True)
        
        highest_score = sorted_diagnoses[0][1]
        possible_diagnoses = [d[0] for d in sorted_diagnoses if d[1] == highest_score]
        
        if highest_score > 0:
            print(f"\nPossible diagnosis: {', '.join(possible_diagnoses).capitalize()}")
            print(f"Confidence score: {highest_score}")
            feedback = input("Was this diagnosis helpful? (yes/no): ").strip().lower()
            if feedback == "yes":
                print("Thank you for your feedback!")
            else:
                print("Thank you! We will work on improving the system.")
        else:
            print("Unable to diagnose. Please consult a healthcare professional.")

if __name__ == "__main__":
    expert_system = MedicalDiagnosisExpertSystem()
    expert_system.diagnose()