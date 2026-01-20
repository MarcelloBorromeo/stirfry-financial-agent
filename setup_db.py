import asyncio
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv
import os

load_dotenv()

DB_CONNECTION = os.environ.get("DB_CONNECTION")

async def main():
    print("🔌 Connecting to Supabase...")
    
    # We use the Async saver just for setup because it has the .setup() method
    async with AsyncPostgresSaver.from_conn_string(DB_CONNECTION) as checkpointer:
        await checkpointer.setup()
        print("✅ Database initialized! Table 'checkpoints' created with correct schema.")

if __name__ == "__main__":
    asyncio.run(main())