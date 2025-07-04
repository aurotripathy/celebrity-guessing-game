import logging
import dspy
import openai
import os   
import sys
from typing import Literal



# voice agent imports
import asyncio
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero, deepgram


logger = logging.getLogger("listen-and-respond")
logger.setLevel(logging.INFO)

class QuestionGenerator(dspy.Signature):
    """Generate a yes/no question in order to guess the celebrity's name in the user's mind.
    You can ask a general question or directly guess the celebrity's name if you think the signal is strong enough
    You should never ask the same question in the past_questions list.
    """
    past_questions: list[str] = dspy.InputField(desc="The list of questions asked in the past")
    past_answers: list[bool] = dspy.InputField(desc="The list of past answers")
    guess_made: bool = dspy.OutputField(desc="If the new_question is a celebrity name guess, set to True, if it is still a general question set to False")
    new_question: str = dspy.OutputField(desc="new question that can help narrow down the celebrity name")


def ask_current_question(prompt, valid_responses=("y", "n")):
    while True:
        response = input(f"{prompt} ({'/'.join(valid_responses)}): ").strip().lower()
        if response in valid_responses:
            return response
        # print(f"Please enter one of: {', '.join(valid_responses)}")

class CelebrityGuess(dspy.Module):
    def __init__(self, max_tries=10):
        super().__init__()
        # Initialize the model with API key
        openai.api_key = os.environ["OPENAI_API_KEY"]
        guessing_llm = dspy.LM('openai/gpt-4o-mini', api_key=openai.api_key)
        dspy.settings.configure(lm=guessing_llm)


    def forward(self):

        class VoiceAgent(Agent):
            def __init__(self) -> None:
                super().__init__(        
                    instructions="",
                    stt=deepgram.STT(),
                    # llm=openai.LLM(model="gpt-4o"),
                    tts=openai.TTS(),
                    vad=silero.VAD.load()
                )
                self.max_tries = 20
                # Initialize the question generator for this voice agent
                self.question_generator = dspy.ChainOfThought(QuestionGenerator)
       
            async def on_enter(self):
                # Start by a one line introduction
                # await self.session.say("We're going to play the game of guess the celebrity.")
                await self.session.say("Please think of a celebrity name.")
                await asyncio.sleep(1)  # Wait 1 second before next question

                # Then ask series of yes/no questions
                await self.ask_series_of_yes_no_questions()
                
                # Set up event handler to listen for user responses
                @self.session.on("user_input_transcribed")
                def on_user_response(event):
                    if event.is_final:
                        logger.info(f'event.transcript: {event.transcript}')
                        asyncio.create_task(self.handle_user_response(event.transcript))
                
            
            
            async def ask_series_of_yes_no_questions(self):
                past_questions = []
                past_answers = []
                correct_guess = False
                for i in range(self.max_tries):
                    question = self.question_generator(
                        past_questions=past_questions,
                        past_answers=past_answers,
                    )
                    await self.session.say(question.new_question)
                    await asyncio.sleep(1)  # Wait 1 second before next question

                    # answer = ask_current_question(f"{question.new_question}").lower() == "y"
                    answer = True # for testing
                    past_questions.append(question.new_question)
                    past_answers.append(answer)

                    if question.guess_made and answer:
                        correct_guess = True
                        break

                if correct_guess:
                    await self.session.say("Yay! I got it right!")
                else:
                    await self.session.say("Oops, I couldn't guess it right.")

                
            
            async def handle_user_response(self, user_text: str):
                logger.info(f'user_text: {user_text}')
                user_text = user_text.lower().strip()
                
                # Respond based on user's answer
                if "yes" in user_text or "yeah" in user_text or "sure" in user_text:
                    await self.session.say("Great!")
                elif "no" in user_text or "nope" in user_text:
                    await self.session.say("Oh, that's too bad.")
                else:
                    await self.session.say("I didn't catch that. Could you answer yes or no?")
                    return  # Don't move to next question if we didn't understand
                
                # Move to next question after a short delay
                await asyncio.sleep(1)  # Wait 1 second before next question

        async def entrypoint(ctx: JobContext):
            await ctx.connect()

            session = AgentSession()

            await session.start(
                agent=VoiceAgent(),
                room=ctx.room
            )

        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

if __name__ == "__main__":
    celebrity_guess = CelebrityGuess()
    celebrity_guess.forward()
