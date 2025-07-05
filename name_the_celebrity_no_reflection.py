import logging
import dspy
import openai
import os

# voice agent imports
import asyncio
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero, deepgram

import logging
loggers = ["LiteLLM Proxy", "LiteLLM Router", "LiteLLM", "httpx"]
for logger_name in loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL + 1) 

logger = logging.getLogger("celebrity_guess")
logger.setLevel(logging.DEBUG)

class QuestionGenerator(dspy.Signature):
    """Generate a yes/no question to guess the celebrity's name in the user's mind.
    You can also directly guess the celebrity's name if you think the signal is strong enough
    You should never ask the same question in the past_questions list.
    """
    past_questions: list[str] = dspy.InputField(desc="The list of questions asked in the past")
    past_answers: list[bool] = dspy.InputField(desc="The list of past answers")
    guess_made: bool = dspy.OutputField(desc="If the new_question is a celebrity name guess, set to True, if it is still a general question set to False")
    new_question: str = dspy.OutputField(desc="new question that can help narrow down the celebrity name")


class CelebrityGuess(dspy.Module):
    """
    This class is a guessor + voice agent that can guess the celebrity's name in the user's mind.
    It uses a question generator to generate questions and a voice agent to ask the questions.
    It uses a voice agent to ask the questions and a voice agent to answer the questions.
    """    
    def __init__(self, max_tries=10):
        super().__init__()
        model = "openai/gpt-4o"
        openai.api_key = os.environ["OPENAI_API_KEY"]
        guessing_llm = dspy.LM(model, api_key=openai.api_key)
        # dspy.settings.configure(lm=guessing_llm)
        dspy.configure(lm=guessing_llm)

    def forward(self):


        class VoiceAgent(Agent):
            def __init__(self) -> None:
                super().__init__(        
                    instructions="",
                    stt=deepgram.STT(),
                    tts=openai.TTS(),
                    vad=silero.VAD.load()
                )
                # Initialize the guessor
                self.max_tries = 20
                self.question_generator = dspy.ChainOfThought(QuestionGenerator)
                self.past_questions = []
                self.past_answers = []
                self.answer = None
                self.guessed_correctly = False
       
            async def on_enter(self):
                # Start by an introduction
                await self.session.say("We're going to play the game of guess the celebrity.")
                await self.session.say("Please think of a celebrity name.")
                await asyncio.sleep(2)  # Wait 2 seconds to give the user time to think of a celebrity name

                @self.session.on("user_input_transcribed")
                def on_transcript(transcript):
                    if transcript.is_final:
                        logger.debug(f'<--Response transcript: {transcript.transcript}')
                        asyncio.create_task(self.handle_user_response(transcript.transcript))  

                # Ask series of yes/no questions
                await self.ask_series_of_yes_no_questions()
                            
            async def ask_series_of_yes_no_questions(self):
                """ Repeat the following until the correct guess is made or the max number of tries is reached. """

                for i in range(self.max_tries):
                    self.question = self.question_generator(
                        past_questions=self.past_questions,
                        past_answers=self.past_answers,
                    )
                    logger.info(f'--> question #{i} from AI: {self.question.new_question}')
                    await self.session.say(self.question.new_question)
                    await asyncio.sleep(3)  # Wait 3 seconds before next question

                    logger.debug(f'++ guess_made: {self.question.guess_made}')
                    if self.question.guess_made and self.answer:
                        self.guessed_correctly = True
                        break

                logger.debug(f'++ Is the guess correct? {self.guessed_correctly}')
                if self.guessed_correctly:
                    await self.session.say("Yay! I guessed right!")
                else:
                    await self.session.say("Oops, I couldn't guess it right.")
                return

                
            
            async def handle_user_response(self, user_text: str):
                
                user_text = user_text.lower().strip()
                logger.info(f'<--User response: {user_text}')
                
                if self.guessed_correctly:
                    return
                
                if "yes" in user_text:
                    self.answer = True
                elif "no" in user_text:
                    self.answer = False
                else:
                    logger.info(f'++ answer unssigned for {user_text}')
                    await self.session.say("I didn't catch that. Could you answer yes or no?")
                    return
                
                logger.debug(f'++ Appending {self.question.new_question} and answer {self.answer}')
                self.past_questions.append(self.question.new_question)
                self.past_answers.append(self.answer)
                logger.debug(f'--> past_questions: {self.past_questions}')
                logger.debug(f'--> past_answers: {self.past_answers}')

                
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
