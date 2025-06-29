import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero, deepgram

# from name_the_celebrity import CelebrityGuess

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

logger = logging.getLogger("listen-and-respond")
logger.setLevel(logging.INFO)



class SimpleAgent(Agent):
    def __init__(self) -> None:
        super().__init__(        
            instructions="",
            stt=deepgram.STT(),
            # llm=openai.LLM(model="gpt-4o"),
            tts=openai.TTS(),
            vad=silero.VAD.load()
        )
        self.current_question_index = 0
        # self.celebrity_guess = CelebrityGuess()

    
    async def on_enter(self):
        # Start by asking the first question
        await self.ask_current_question()
        
        # Set up event handler to listen for user responses
        @self.session.on("user_input_transcribed")
        def on_user_response(event):
            if event.is_final:
                asyncio.create_task(self.handle_user_response(event.transcript))
    
    async def ask_current_question(self):
        # await self.celebrity_guess.forward()

        if self.current_question_index < len(yes_no_questions):
            question = yes_no_questions[self.current_question_index]
            await self.session.say(question)
        else:
            await self.session.say("That's all the questions! Thank you for answering.")
    
    async def handle_user_response(self, user_text: str):
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
        self.current_question_index += 1
        await self.ask_current_question()

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession()

    await session.start(
        agent=SimpleAgent(),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))