class PlantHealthExpertSystem:
    def __init__(self):
        self.rules = {
            "underwatered": [
                ("Is the soil dry?", "yes", 2),
                ("Are the leaves wilting?", "yes", 3),
            ],
            "overwatered": [
                ("Is the soil wet?", "yes", 3),
                ("Are the leaves yellowing?", "yes", 2),
                ("Is there mold on the surface?", "yes", 1),
            ],
            "pests": [
                ("Do you see any bugs on the plant?", "yes", 3),
                ("Are the leaves damaged?", "yes", 2),
                ("Is there webbing on the leaves?", "yes", 1),
            ],
            "healthy": [
                ("Is the soil moist?", "yes", 3),
                ("Are the leaves green and firm?", "yes", 3),
                ("Are there any new shoots or flowers?", "yes", 2),
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
        
        # Sort diagnoses by score
        sorted_diagnoses = sorted(diagnosis_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Display results
        highest_score = sorted_diagnoses[0][1]
        possible_diagnoses = [d[0] for d in sorted_diagnoses if d[1] == highest_score]
        
        if highest_score > 0:
            print(f"\nThe plant may be: {', '.join(possible_diagnoses).capitalize()}")
            print(f"Confidence score: {highest_score}")
            feedback = input("Was this diagnosis helpful? (yes/no): ").strip().lower()
            if feedback == "yes":
                print("Thank you for your feedback!")
            else:
                print("Thank you! We will work on improving the system.")
        else:
            print("Unable to diagnose the plant's health. Please consult an expert.")

if __name__ == "__main__":
    expert_system = PlantHealthExpertSystem()
    expert_system.diagnose()
