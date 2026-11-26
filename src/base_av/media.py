"""Read and write release MP4 files."""
from pathlib import Path
from tempfile import NamedTemporaryFile


def complete_video(path, frames, *, size=None, fps=None):
    if not path.is_file():
        return False
    import av
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if size is not None and (stream.width, stream.height) != size:
                return False
            if fps is not None and stream.average_rate != fps:
                return False
            return sum(1 for _ in container.decode(stream)) == frames
    except (ValueError, OSError, IndexError):
        return False


def save_video(video, path, frames, fps):
    import av
    if video.shape[1] < frames:
        raise ValueError(f'Decoded {video.shape[1]} frames, expected {frames}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix='.mp4', delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        pixels = video[:, :frames].permute(1, 2, 3, 0).cpu().numpy()
        with av.open(str(temporary), 'w') as container:
            stream = container.add_stream('libx264', rate=fps)
            stream.width, stream.height = pixels.shape[2], pixels.shape[1]
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '10'}
            for rgb in pixels:
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        if not complete_video(temporary, frames):
            raise OSError(f'Could not verify {frames} frames in {temporary}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
