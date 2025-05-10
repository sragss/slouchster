from ollama import chat
import re
import statistics
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Define constants
MODEL = "gemma3:27b-it-qat"
image_paths = ["test_imgs/good.jpg", "test_imgs/bad.jpg"]
results = {"good": {"shoulder": [], "spine": []}, "bad": {"shoulder": [], "spine": []}}

# Function to extract scores from response
def extract_scores(response_text):
    shoulder_match = re.search(r"Shoulder Position: (\d+)", response_text)
    spine_match = re.search(r"Spine Alignment: (\d+)", response_text)
    
    shoulder_score = int(shoulder_match.group(1)) if shoulder_match else None
    spine_score = int(spine_match.group(1)) if spine_match else None
    
    return shoulder_score, spine_score

# Run 4 times for each image
console = Console()
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console
) as progress:
    for img_path in image_paths:
        img_type = "good" if "good" in img_path else "bad"
        
        for i in range(4):
            task = progress.add_task(f"Processing {img_type} image (run {i+1}/4)...", total=None)
            
            response = chat(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": "You are a posture assessment system analyzing webcam images. Evaluate the following aspects of posture:\n\n1. Shoulder Position (0-100): 0 means severely slouched shoulders, 100 means perfectly aligned shoulders.\n2. Spine Alignment (0-100): 0 means severely hunched/curved spine, 100 means ideal vertical alignment.\n\nProvide your assessment in this exact format only:\nShoulder Position: [score]\nSpine Alignment: [score]\n\nDo not include any other text, explanations, or commentary in your response.",
                        "images": [img_path]
                    }
                ]
            )
            
            progress.update(task, completed=True)
            
            shoulder_score, spine_score = extract_scores(response["message"]["content"])
            if shoulder_score is not None:
                results[img_type]["shoulder"].append(shoulder_score)
            if spine_score is not None:
                results[img_type]["spine"].append(spine_score)

# Calculate statistics and display in a rich table
table = Table(title="Posture Assessment Results")

table.add_column("Image Type", style="cyan")
table.add_column("Metric", style="magenta")
table.add_column("Average", style="green")
table.add_column("Std Dev", style="yellow")

for img_type in ["good", "bad"]:
    for metric, scores in results[img_type].items():
        if scores:
            avg = statistics.mean(scores)
            stdev = statistics.stdev(scores) if len(scores) > 1 else 0
            table.add_row(
                img_type.capitalize(), 
                metric.capitalize(), 
                f"{avg:.2f}", 
                f"{stdev:.2f}"
            )

console.print(table)
