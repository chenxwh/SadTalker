"""run bash scripts/download_models.sh first to prepare the weights file"""

import os
import shutil
import subprocess
import time
import torch
from cog import BasePredictor, Input, Path
from pydub import AudioSegment

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path


# prepare weights from https://huggingface.co/spaces/vinthony/SadTalker/tree/main/checkpoints, then push and load from replicate.delivery for faster inference
MODEL_URL = (
    "https://weights.replicate.delivery/default/vinthony/SadTalker/checkpoints.tar"
)
GFPGA_URL = "https://weights.replicate.delivery/default/vinthony/SadTalker/gfpgan.tar"
MODEL_CACHE = "checkpoints"
GFPGAN_CACHE = "gfpgan"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        if not os.path.exists(GFPGAN_CACHE):
            download_weights(GFPGA_URL, GFPGAN_CACHE)
        self.sad_talker = SadTalker(checkpoint_path=MODEL_CACHE)

    def predict(
        self,
        source_image: Path = Input(
            description="Upload the source image, it can be video.mp4 or picture.png",
        ),
        driven_audio: Path = Input(
            description="Upload the driven audio, accepts .wav and .mp4 file",
        ),
        use_enhancer: bool = Input(
            description="Use GFPGAN as Face enhancer",
            default=False,
        ),
        pose_style: int = Input(description="Pose style", le=45, ge=0, default=0),
        expression_scale: float = Input(
            description=" a larger value will make the expression motion stronger",
            default=1.0,
        ),
        use_eyeblink: bool = Input(
            description="Use eye blink",
            default=True,
        ),
        preprocess: str = Input(
            description="Choose how to preprocess the images",
            choices=["crop", "resize", "full", "extcrop", "extfull"],
            default="crop",
        ),
        size_of_image: int = Input(
            description="Face model resolution", choices=[256, 512], default=256
        ),
        facerender: str = Input(
            description="Choose face render",
            choices=["facevid2vid", "pirender"],
            default="facevid2vid",
        ),
        still_mode: bool = Input(
            description="Still Mode (fewer head motion, works with preprocess 'full')",
            default=True,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        exp_dir = "exp_dir"
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        os.makedirs(exp_dir)

        mp4_path = self.sad_talker.test(
            source_image=str(source_image),
            driven_audio=str(driven_audio),
            preprocess=preprocess,
            still_mode=still_mode,
            use_enhancer=use_enhancer,
            batch_size=1,
            size=size_of_image,
            pose_style=pose_style,
            exp_scale=expression_scale,
            use_ref_video=False,
            ref_video=None,
            ref_info=None,
            length_of_audio=0,
            use_blink=use_eyeblink,
            save_dir=exp_dir,
        )

        output = "/tmp/out.mp4"
        shutil.copy(mp4_path, output)
        return Path(output)


def mp3_to_wav(mp3_filename, wav_filename, frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename, format="wav")


class SadTalker:

    def __init__(self, checkpoint_path="checkpoints", config_path="src/config"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["TORCH_HOME"] = checkpoint_path
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path

    def test(
        self,
        source_image,
        driven_audio,
        preprocess="crop",
        still_mode=False,
        use_enhancer=False,
        batch_size=1,
        size=256,
        pose_style=0,
        exp_scale=1.0,
        use_ref_video=False,
        ref_video=None,
        ref_info=None,
        length_of_audio=0,
        use_blink=True,
        save_dir="./exp_dir/",
    ):

        self.sadtalker_paths = init_path(
            self.checkpoint_path, self.config_path, size, False, preprocess
        )

        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)
        self.animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.device)

        if ".mp3" in driven_audio:
            audio_path = "input.wav"
            mp3_to_wav(driven_audio, audio_path, 16000)
        else:
            audio_path = driven_audio

        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, "first_frame_dir")
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
            source_image, first_frame_dir, preprocess, True, size
        )

        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        ref_video_coeff_path = None
        ref_pose_coeff_path = None
        ref_eyeblink_coeff_path = None

        batch = get_data(
            first_coeff_path,
            audio_path,
            self.device,
            ref_eyeblink_coeff_path=ref_eyeblink_coeff_path,
            still=still_mode,
            idlemode=False,
            length_of_audio=length_of_audio,
            use_blink=use_blink,
        )  # longer audio?
        coeff_path = self.audio_to_coeff.generate(
            batch, save_dir, pose_style, ref_pose_coeff_path
        )

        # coeff2video
        data = get_facerender_data(
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            audio_path,
            batch_size,
            still_mode=still_mode,
            preprocess=preprocess,
            size=size,
            expression_scale=exp_scale,
        )
        return_path = self.animate_from_coeff.generate(
            data,
            save_dir,
            source_image,
            crop_info,
            enhancer="gfpgan" if use_enhancer else None,
            preprocess=preprocess,
            img_size=size,
        )
        video_name = data["video_name"]
        print(f"The generated video is named {video_name} in {save_dir}")

        del self.preprocess_model
        del self.audio_to_coeff
        del self.animate_from_coeff

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        import gc

        gc.collect()

        return return_path
