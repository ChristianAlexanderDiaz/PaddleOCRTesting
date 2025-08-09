import os
import gc
import threading
import asyncio
import aiohttp
import tempfile
from dotenv import load_dotenv
import discord
from discord import app_commands
from discord.ext import commands
from paddleocr import PaddleOCR
from PIL import Image
import traceback

# Load environment variables
load_dotenv()

# Thread lock for OCR operations
ocr_lock = threading.Lock()

# Ultra-light initialization (sacrifices some accuracy for stability)
ocr = PaddleOCR(
    det_model_name='en_PP-OCRv3_det_slim',  # Slim detection model
    rec_model_name='en_PP-OCRv3_rec_slim',  # Slim recognition model
    use_angle_cls=False,
    lang='en',
    use_gpu=False,
    show_log=False,
    det_db_thresh=0.3,
    det_db_box_thresh=0.6,
    max_text_length=25,
    rec_batch_num=1,
    cpu_threads=1,
    enable_mkldnn=False,  # Disable MKL-DNN to save memory
    det_limit_side_len=640,  # Smaller image processing
    rec_image_shape="3, 32, 320"  # Even smaller recognition
)

print("PaddleOCR initialized!")

def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()

class OCRBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)

    async def setup_hook(self):
        try:
            synced = await self.tree.sync()
            print(f"Synced {len(synced)} command(s)")
        except Exception as e:
            print(f"Failed to sync commands: {e}")

    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')
        print(f'Bot is in {len(self.guilds)} guilds')

bot = OCRBot()

async def download_image(url: str) -> str:
    """Download image from URL and return temporary file path"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to download image: HTTP {response.status}")
            
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            try:
                with os.fdopen(temp_fd, 'wb') as temp_file:
                    temp_file.write(await response.read())
                return temp_path
            except:
                os.close(temp_fd)
                raise

def perform_ocr_on_file(image_path: str) -> dict:
    """Perform OCR on image file and return results"""
    try:
        with ocr_lock:
            # Perform OCR
            result = ocr.ocr(image_path, cls=False)
            
            # Format results
            text_results = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:  # Ensure valid structure
                        text_results.append({
                            "text": line[1][0],
                            "confidence": float(line[1][1]),
                            "bbox": line[0]
                        })
            
            response = {
                "success": True,
                "results": text_results,
                "text": " ".join([r["text"] for r in text_results])
            }
            
            # Clean up
            del result
            cleanup_memory()
            
            return response
            
    except Exception as e:
        cleanup_memory()
        print(f"Error in OCR: {str(e)}")
        print(traceback.format_exc())
        return {"success": False, "error": str(e)}

async def find_recent_image(interaction: discord.Interaction, limit: int = 50):
    """Find the most recent image in the channel"""
    try:
        async for message in interaction.channel.history(limit=limit):
            # Check attachments
            for attachment in message.attachments:
                if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                    return attachment.url, f"attachment: {attachment.filename}"
            
            # Check embeds
            for embed in message.embeds:
                if embed.image and embed.image.url:
                    return embed.image.url, "embed image"
                if embed.thumbnail and embed.thumbnail.url:
                    return embed.thumbnail.url, "embed thumbnail"
        
        return None, None
    except Exception as e:
        print(f"Error finding recent image: {e}")
        return None, None

@bot.tree.command(name='runocr', description='Run OCR on the most recent image in this channel')
async def runocr(interaction: discord.Interaction):
    """Run OCR on the most recent image in the channel"""
    await interaction.response.defer(thinking=True)
    
    try:
        # Find recent image
        image_url, image_source = await find_recent_image(interaction)
        
        if not image_url:
            await interaction.followup.send("❌ No recent images found in this channel (checked last 50 messages)")
            return
        
        # Download image
        try:
            temp_path = await download_image(image_url)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to download image: {str(e)}")
            return
        
        try:
            # Perform OCR
            ocr_result = perform_ocr_on_file(temp_path)
            
            if not ocr_result["success"]:
                await interaction.followup.send(f"❌ OCR failed: {ocr_result.get('error', 'Unknown error')}")
                return
            
            # Format results for Discord
            if not ocr_result["text"].strip():
                await interaction.followup.send("✅ OCR completed, but no text was detected in the image.")
                return
            
            # Create embed for results
            embed = discord.Embed(
                title="🔍 OCR Results",
                color=0x00ff00,
                description=f"**Source:** {image_source}\n**Detected Text:**"
            )
            
            # Truncate text if too long for Discord
            text_content = ocr_result["text"]
            if len(text_content) > 4000:
                text_content = text_content[:4000] + "...\n*(text truncated)*"
            
            embed.add_field(
                name="📝 Extracted Text",
                value=f"```\n{text_content}\n```",
                inline=False
            )
            
            # Add confidence info if available
            if ocr_result["results"]:
                avg_confidence = sum(r["confidence"] for r in ocr_result["results"]) / len(ocr_result["results"])
                embed.add_field(
                    name="📊 Stats",
                    value=f"Lines detected: {len(ocr_result['results'])}\nAvg confidence: {avg_confidence:.2f}",
                    inline=True
                )
            
            await interaction.followup.send(embed=embed)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        print(f"Error in runocr command: {e}")
        print(traceback.format_exc())
        await interaction.followup.send(f"❌ An error occurred: {str(e)}")

# Periodic garbage collection
async def periodic_cleanup():
    while True:
        await asyncio.sleep(30)  # Every 30 seconds
        cleanup_memory()

@bot.event
async def on_ready():
    # Start cleanup task
    asyncio.create_task(periodic_cleanup())

if __name__ == '__main__':
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        print("Error: DISCORD_BOT_TOKEN environment variable not found!")
        print("Please set your Discord bot token in the .env file")
        exit(1)
    
    print("Starting Discord OCR Bot...")
    bot.run(token)