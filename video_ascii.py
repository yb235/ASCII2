#!/usr/bin/env python3
"""
VIDEO ASCII ANIMATOR - Convert Videos to Animated ASCII Art!

Features:
- Process video files frame by frame
- Generate animated ASCII sequences
- Multiple output formats (text frames, HTML animation, ANSI animation)
- Adaptive frame sampling for smooth playback
- Color support for stunning animated ASCII
- Real-time terminal playback

The ultimate video ASCII experience!
"""

import os
import sys
import time
import subprocess
from typing import List, Tuple, Optional, Generator
from PIL import Image, ImageSequence
import json
from ultimate_ascii import UltimateASCIIConverter


class VideoASCIIAnimator:
    """
    Convert videos to animated ASCII art with multiple output formats.
    """
    
    def __init__(self, ascii_converter: UltimateASCIIConverter = None,
                 frame_skip: int = 1, max_frames: int = 300):
        """
        Initialize the video ASCII animator.
        
        Args:
            ascii_converter: UltimateASCIIConverter instance
            frame_skip: Process every Nth frame (1 = all frames)
            max_frames: Maximum frames to process
        """
        self.ascii_converter = ascii_converter or UltimateASCIIConverter(
            width=120, char_set='blocks_4x', use_color=True
        )
        self.frame_skip = frame_skip
        self.max_frames = max_frames
    
    def extract_video_frames(self, video_path: str, output_dir: str = "temp_frames") -> List[str]:
        """
        Extract frames from video using ffmpeg (if available) or PIL.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to store extracted frames
            
        Returns:
            List of frame file paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to use ffmpeg for better video support
        try:
            # Check if ffmpeg is available
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            
            # Extract frames with ffmpeg
            frame_pattern = os.path.join(output_dir, "frame_%04d.png")
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', f'fps=1/{self.frame_skip}',  # Sample frames
                '-y',  # Overwrite existing files
                frame_pattern
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Get list of generated frames
            frame_files = []
            for i in range(1, self.max_frames + 1):
                frame_file = os.path.join(output_dir, f"frame_{i:04d}.png")
                if os.path.exists(frame_file):
                    frame_files.append(frame_file)
                else:
                    break
            
            return frame_files
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ ffmpeg not found, trying PIL for GIF support...")</abstract>
            
            # Fallback to PIL for GIF files
            if not video_path.lower().endswith('.gif'):
                raise ValueError("Without ffmpeg, only GIF files are supported")
            
            return self.extract_gif_frames(video_path, output_dir)
    
    def extract_gif_frames(self, gif_path: str, output_dir: str) -> List[str]:
        """Extract frames from GIF using PIL."""
        frame_files = []
        
        try:
            with Image.open(gif_path) as img:
                frame_count = 0
                for frame_num, frame in enumerate(ImageSequence.Iterator(img)):
                    if frame_num % self.frame_skip == 0 and frame_count < self.max_frames:
                        frame_file = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
                        frame.convert('RGB').save(frame_file)
                        frame_files.append(frame_file)
                        frame_count += 1
                        
        except Exception as e:
            raise ValueError(f"Could not process GIF: {e}")
        
        return frame_files
    
    def convert_frames_to_ascii(self, frame_files: List[str]) -> List[Tuple[List[str], Optional[List]]]:
        """
        Convert video frames to ASCII art.
        
        Args:
            frame_files: List of frame image paths
            
        Returns:
            List of (ascii_lines, color_data) tuples
        """
        ascii_frames = []
        total_frames = len(frame_files)
        
        print(f"🎬 Converting {total_frames} frames to ASCII...")
        
        for i, frame_file in enumerate(frame_files):
            try:
                # Load frame
                frame_image = Image.open(frame_file)
                
                # Convert to ASCII
                ascii_lines, color_data = self.ascii_converter.convert_to_8k_ascii(frame_image)
                ascii_frames.append((ascii_lines, color_data))
                
                # Progress indicator
                if (i + 1) % 10 == 0 or i == total_frames - 1:
                    progress = (i + 1) / total_frames * 100
                    print(f"📊 Progress: {progress:.1f}% ({i + 1}/{total_frames} frames)")
                    
            except Exception as e:
                print(f"⚠️ Error processing frame {i}: {e}")
                continue
        
        print(f"✅ Successfully converted {len(ascii_frames)} frames!")
        return ascii_frames
    
    def save_text_animation(self, ascii_frames: List[Tuple[List[str], Optional[List]]],
                           output_path: str, fps: float = 10.0) -> None:
        """
        Save ASCII animation as text file with frame separators.
        
        Args:
            ascii_frames: List of ASCII frame data
            output_path: Output file path
            fps: Frames per second for playback timing
        """
        frame_delay = 1.0 / fps
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# ASCII Video Animation\n")
            f.write(f"# Total frames: {len(ascii_frames)}\n")
            f.write(f"# FPS: {fps}\n")
            f.write(f"# Frame delay: {frame_delay:.3f}s\n")
            f.write(f"# Use video_ascii_player.py to play this file\n\n")
            
            for frame_num, (ascii_lines, color_data) in enumerate(ascii_frames):
                f.write(f"FRAME:{frame_num:04d}\n")
                f.write('\n'.join(ascii_lines))
                f.write('\n\nFRAME_END\n\n')
        
        print(f"💾 Text animation saved to: {output_path}")
    
    def save_html_animation(self, ascii_frames: List[Tuple[List[str], Optional[List]]],
                           output_path: str, fps: float = 10.0) -> None:
        """
        Save ASCII animation as interactive HTML with JavaScript playback.
        
        Args:
            ascii_frames: List of ASCII frame data
            output_path: Output HTML file path
            fps: Frames per second
        """
        frame_delay = int(1000 / fps)  # milliseconds
        
        # Prepare frame data for JavaScript
        frames_data = []
        for ascii_lines, color_data in ascii_frames:
            if self.ascii_converter.use_color and color_data:
                # Create colored HTML for each frame
                frame_html = ""
                for row, line in enumerate(ascii_lines):
                    line_html = ""
                    for col, char in enumerate(line):
                        if row < len(color_data) and col < len(color_data[row]):
                            r, g, b = color_data[row][col]
                            line_html += f'<span style="color:rgb({r},{g},{b})">{char}</span>'
                        else:
                            line_html += char
                    frame_html += line_html + '\\n'
                frames_data.append(frame_html)
            else:
                # Monochrome frame
                frames_data.append('\\n'.join(ascii_lines).replace('"', '\\"'))
        
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 ASCII Video Animation</title>
    <style>
        body {{
            background: #000;
            color: #fff;
            font-family: "Fira Code", "Consolas", "Courier New", monospace;
            margin: 0;
            padding: 20px;
            overflow: hidden;
        }}
        .container {{
            text-align: center;
        }}
        .ascii-screen {{
            background: #111;
            border: 2px solid #00ff00;
            border-radius: 10px;
            padding: 20px;
            margin: 20px auto;
            display: inline-block;
            box-shadow: 0 0 30px rgba(0, 255, 0, 0.3);
        }}
        .ascii-frame {{
            font-size: 1px;
            line-height: 1px;
            letter-spacing: 0;
            white-space: pre;
            font-weight: normal;
            text-align: left;
        }}
        .controls {{
            margin: 20px;
        }}
        .btn {{
            background: #333;
            color: #fff;
            border: 1px solid #555;
            padding: 10px 20px;
            margin: 5px;
            cursor: pointer;
            border-radius: 5px;
            font-family: inherit;
        }}
        .btn:hover {{
            background: #555;
        }}
        .info {{
            color: #888;
            font-size: 14px;
            margin: 10px;
        }}
        .title {{
            color: #00ff00;
            text-shadow: 0 0 10px #00ff00;
            font-size: 24px;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">🎬 ASCII VIDEO ANIMATION 🎬</h1>
        <div class="info">
            Frames: {len(ascii_frames)} | FPS: {fps} | Resolution: {len(ascii_frames[0][0][0])}×{len(ascii_frames[0][0])}
        </div>
        
        <div class="ascii-screen">
            <div id="ascii-frame" class="ascii-frame"></div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="playPause()">▶️ Play/Pause</button>
            <button class="btn" onclick="restart()">⏮️ Restart</button>
            <button class="btn" onclick="prevFrame()">⏪ Previous</button>
            <button class="btn" onclick="nextFrame()">⏩ Next</button>
            <button class="btn" onclick="changeSpeed(0.5)">🐌 0.5x</button>
            <button class="btn" onclick="changeSpeed(1)">🚶 1x</button>
            <button class="btn" onclick="changeSpeed(2)">🏃 2x</button>
        </div>
        
        <div class="info">
            Frame: <span id="frame-counter">1</span>/{len(ascii_frames)} | 
            Speed: <span id="speed-indicator">1x</span>
        </div>
    </div>

    <script>
        const frames = {json.dumps(frames_data)};
        let currentFrame = 0;
        let isPlaying = false;
        let animationInterval;
        let currentSpeed = 1;
        let frameDelay = {frame_delay};
        
        const frameElement = document.getElementById('ascii-frame');
        const frameCounter = document.getElementById('frame-counter');
        const speedIndicator = document.getElementById('speed-indicator');
        
        function showFrame(frameIndex) {{
            if (frameIndex >= 0 && frameIndex < frames.length) {{
                frameElement.innerHTML = frames[frameIndex];
                frameCounter.textContent = frameIndex + 1;
                currentFrame = frameIndex;
            }}
        }}
        
        function playPause() {{
            if (isPlaying) {{
                clearInterval(animationInterval);
                isPlaying = false;
            }} else {{
                animationInterval = setInterval(() => {{
                    currentFrame = (currentFrame + 1) % frames.length;
                    showFrame(currentFrame);
                }}, frameDelay / currentSpeed);
                isPlaying = true;
            }}
        }}
        
        function restart() {{
            currentFrame = 0;
            showFrame(currentFrame);
        }}
        
        function nextFrame() {{
            currentFrame = (currentFrame + 1) % frames.length;
            showFrame(currentFrame);
        }}
        
        function prevFrame() {{
            currentFrame = (currentFrame - 1 + frames.length) % frames.length;
            showFrame(currentFrame);
        }}
        
        function changeSpeed(speed) {{
            currentSpeed = speed;
            speedIndicator.textContent = speed + 'x';
            if (isPlaying) {{
                playPause();
                playPause();
            }}
        }}
        
        // Initialize
        showFrame(0);
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {{
            switch(e.key) {{
                case ' ': playPause(); break;
                case 'ArrowRight': nextFrame(); break;
                case 'ArrowLeft': prevFrame(); break;
                case 'r': restart(); break;
            }}
        }});
    </script>
</body>
</html>'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🌐 Interactive HTML animation saved to: {output_path}")
        print(f"🎮 Open in browser for interactive playback with controls!")
    
    def create_ansi_player_script(self, ascii_frames: List[Tuple[List[str], Optional[List]]],
                                 output_path: str, fps: float = 10.0) -> None:
        """
        Create a Python script for terminal ANSI animation playback.
        
        Args:
            ascii_frames: List of ASCII frame data  
            output_path: Output Python script path
            fps: Frames per second
        """
        frame_delay = 1.0 / fps
        
        script_content = f'''#!/usr/bin/env python3
"""
ANSI ASCII Video Player - Terminal Animation Playback
Generated by Ultimate ASCII Video Animator

Press Ctrl+C to stop playback
"""

import time
import sys
import os

# Animation data
FRAME_DELAY = {frame_delay}
FRAMES = ['''
        
        for i, (ascii_lines, color_data) in enumerate(ascii_frames):
            if self.ascii_converter.use_color and color_data:
                # Generate ANSI colored frame
                ansi_frame = self.ascii_converter.format_ansi_output(ascii_lines, color_data)
                script_content += f'    """{ansi_frame}""",\n'
            else:
                # Plain text frame
                frame_text = '\\n'.join(ascii_lines).replace('"""', '\\"""')
                script_content += f'    """{frame_text}""",\n'
        
        script_content += f''']

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def play_animation():
    """Play the ASCII animation in terminal."""
    try:
        frame_count = len(FRAMES)
        print(f"🎬 Playing ASCII animation: {{frame_count}} frames at {{1/FRAME_DELAY:.1f}} FPS")
        print("Press Ctrl+C to stop\\n")
        time.sleep(2)
        
        while True:
            for i, frame in enumerate(FRAMES):
                clear_screen()
                print(f"Frame {{i+1}}/{{frame_count}}")
                print(frame)
                time.sleep(FRAME_DELAY)
                
    except KeyboardInterrupt:
        clear_screen()
        print("🛑 Animation stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    play_animation()
'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        if os.name != 'nt':
            os.chmod(output_path, 0o755)
        
        print(f"📺 ANSI player script saved to: {output_path}")
        print(f"▶️ Run with: python {output_path}")
    
    def process_video(self, video_path: str, output_prefix: str = "ascii_video",
                     fps: float = 10.0, cleanup_frames: bool = True) -> Dict[str, str]:
        """
        Process a complete video to ASCII animation in multiple formats.
        
        Args:
            video_path: Input video file path
            output_prefix: Prefix for output files
            fps: Target frames per second
            cleanup_frames: Whether to delete temporary frame files
            
        Returns:
            Dictionary of output file paths
        """
        print(f"🎬 Processing video: {video_path}")
        print(f"⚙️ Settings: {self.ascii_converter.width}×{int(self.ascii_converter.width * 0.56)} | {fps} FPS | Skip: {self.frame_skip}")
        
        # Extract frames
        temp_dir = f"temp_frames_{int(time.time())}"
        frame_files = self.extract_video_frames(video_path, temp_dir)
        
        if not frame_files:
            raise ValueError("No frames could be extracted from video")
        
        # Convert frames to ASCII
        ascii_frames = self.convert_frames_to_ascii(frame_files)
        
        # Generate outputs
        outputs = {}
        
        # Text animation
        text_output = f"{output_prefix}.txt"
        self.save_text_animation(ascii_frames, text_output, fps)
        outputs['text'] = text_output
        
        # HTML animation  
        html_output = f"{output_prefix}.html"
        self.save_html_animation(ascii_frames, html_output, fps)
        outputs['html'] = html_output
        
        # ANSI terminal player
        ansi_output = f"{output_prefix}_player.py"
        self.create_ansi_player_script(ascii_frames, ansi_output, fps)
        outputs['ansi'] = ansi_output
        
        # Cleanup temporary files
        if cleanup_frames:
            for frame_file in frame_files:
                try:
                    os.remove(frame_file)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
            print(f"🧹 Cleaned up temporary files")
        
        return outputs


def create_video_ascii(video_path: str, output_prefix: str = "ascii_video",
                      width: int = 120, fps: float = 10.0, 
                      char_set: str = 'blocks_4x', use_color: bool = True,
                      frame_skip: int = 1, max_frames: int = 300) -> Dict[str, str]:
    """
    Create ASCII animation from video file.
    
    Args:
        video_path: Input video file
        output_prefix: Output files prefix  
        width: ASCII width
        fps: Target FPS
        char_set: Character set style
        use_color: Whether to use color
        frame_skip: Process every Nth frame
        max_frames: Maximum frames to process
        
    Returns:
        Dictionary of generated output files
    """
    # Create ASCII converter
    converter = UltimateASCIIConverter(
        width=width,
        char_set=char_set,
        use_color=use_color,
        output_format='ansi' if use_color else 'text'
    )
    
    # Create animator
    animator = VideoASCIIAnimator(
        ascii_converter=converter,
        frame_skip=frame_skip,
        max_frames=max_frames
    )
    
    # Process video
    return animator.process_video(video_path, output_prefix, fps)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🎬 VIDEO ASCII ANIMATOR - Convert videos to animated ASCII art! 🎬",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert GIF to ASCII animation
    python video_ascii.py animation.gif
    
    # High quality colored ASCII video  
    python video_ascii.py video.mp4 -w 200 -c --fps 15
    
    # Fast preview (every 3rd frame)
    python video_ascii.py video.mp4 --skip 3 --max-frames 100
        """
    )
    
    parser.add_argument('video', help='Input video file (MP4, GIF, etc.)')
    parser.add_argument('-o', '--output', default='ascii_video', help='Output prefix')
    parser.add_argument('-w', '--width', type=int, default=120, help='ASCII width')
    parser.add_argument('--fps', type=float, default=10.0, help='Output FPS')
    parser.add_argument('-s', '--char-set', default='blocks_4x', help='Character set')
    parser.add_argument('-c', '--color', action='store_true', help='Use color')
    parser.add_argument('--skip', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--max-frames', type=int, default=300, help='Maximum frames')
    
    args = parser.parse_args()
    
    try:
        outputs = create_video_ascii(
            args.video, args.output, args.width, args.fps,
            args.char_set, args.color, args.skip, args.max_frames
        )
        
        print(f"\n🎉 VIDEO ASCII ANIMATION COMPLETE! 🎉")
        print("=" * 50)
        for format_type, file_path in outputs.items():
            print(f"{format_type.upper()}: {file_path}")
        
        print(f"\n🎮 PLAYBACK OPTIONS:")
        print(f"🌐 HTML: Open {outputs['html']} in browser")
        print(f"📺 Terminal: python {outputs['ansi']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)