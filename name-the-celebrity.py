
import dspy
import openai
import os   
import sys
from typing import Literal

openai.api_key = os.environ["OPENAI_API_KEY"]
lm = dspy.LM('openai/gpt-4o-mini', api_key=openai.api_key)
dspy.settings.configure(lm=lm)
dspy.inspect_history(n=1)

# Initialize the model with API key
try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    lm = dspy.LM('openai/gpt-4o-mini', api_key=openai.api_key)
    dspy.settings.configure(lm=lm)
    dspy.inspect_history(n=1)
except Exception as e:
    print(f"Error initializing OpenAI model: {str(e)}")
    sys.exit(1)

class QuestionGenerator(dspy.Signature):
    """Generate a yes/no question in order to guess the celebrity's name in the user's mind.
    You can ask a general question or directly guess the celebrity's name if you think the signal is strong enough
    You should never ask the same question in the past_questions list.
    """
    past_questions: list[str] = dspy.InputField(desc="The list of questions asked in the past")
    past_answers: list[bool] = dspy.InputField(desc="The list of past answers")
    guess_made: bool = dspy.OutputField(desc="If the new_question is a celebrity name guess, set to True, if it is still a general question set to False")
    new_question: str = dspy.OutputField(desc="new question that can help narrow down the celebrity name")

class Reflection(dspy.Signature):
    """Provide reflection on the guessing process"""
    correct_celebrity_name: str = dspy.InputField(desc="The celebrity name in the user's mind")
    final_guessor_question: str = dspy.InputField(desc="the final guess or question the LM made")
    past_questions: list[str] = dspy.InputField(desc="The list of questions asked in the past")
    past_answers: list[bool] = dspy.InputField(desc="The list of past answers")
    reflection: str = dspy.OutputField(desc="reflection on the guessing process, including what was done well and what can be improved")

def ask(prompt, valid_responses=("y", "n")):
    while True:
        response = input(f"{prompt} ({'/'.join(valid_responses)}): ").strip().lower()
        if response in valid_responses:
            return response
        print(f"Please enter one of: {', '.join(valid_responses)}")

class CelebrityGuess(dspy.Module):
    def __init__(self, max_tries=10):
        super().__init__()

        self.question_generator = dspy.ChainOfThought(QuestionGenerator)
        self.reflection = dspy.ChainOfThought(Reflection)

        self.max_tries = 20

    def forward(self):
        celebrity_name = input("Please think of a celebrity name, when ready, type the name and press enter...")
        past_questions = []
        past_answers = []

        correct_guess = False

        for i in range(self.max_tries):
            question = self.question_generator(
                past_questions=past_questions,
                past_answers=past_answers,
            )
            answer = ask(f"{question.new_question}").lower() == "y"
            past_questions.append(question.new_question)
            past_answers.append(answer)

            if question.guess_made and answer:
                correct_guess = True
                break

        if correct_guess:
            print("Yay! I got it right!")
        else:
            print("Oops, I couldn't guess it right.")

        reflection = self.reflection(
            correct_celebrity_name=celebrity_name,
            final_guessor_question=question.new_question,
            past_questions=past_questions,
            past_answers=past_answers,
        )
        print(reflection.reflection)



if __name__ == "__main__":
    celebrity_guess = CelebrityGuess()
    celebrity_guess()
