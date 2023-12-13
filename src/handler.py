'''
Handler for the generation of a fine tuned lora model.
'''

import os
import shutil
import subprocess
import time

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_download, upload_file_to_bucket

from hurry.filesize import size
from rp_schema import INPUT_SCHEMA

print_json_output = []
def print_json(data):
    print_json_output.append(data)
    print(data)

def list_disk_usage(path, do_walk=False):
    path_bytes = os.path.getsize(path)
    path_size = size(path_bytes)
    path_stat = shutil.disk_usage(path) 

    list = []
    list.append(path + " (" + path_size + ") USAGE total:" + size(path_stat.total) + " used:" + size(path_stat.used) + " free:" + size(path_stat.free) + ")")

    if do_walk:
        for root, dirs, files in os.walk(path):
            for name in files:
                name_path = os.path.join(root, name)
                name_bytes = os.path.getsize(name_path)
                name_size = size(name_bytes)
                list.append(name_path + " (" + name_size + ")")

            for name in dirs:
                name_path = os.path.join(root, name)
                name_bytes = os.path.getsize(name_path)
                name_size = size(name_bytes)
                list.append(name_path + " (" + name_size + ")")

    return list



def collect_disc_usage():
    runpod_volume_usage = list_disk_usage(runpod_volume_path, True)
    workspage_usage = [] # list_disk_usage(os.getcwd(), True)
    files = runpod_volume_usage + workspage_usage
    for file in files:
        print(file)        
    
    return files

def handler(job):

    print_json("handler()")


    # Clear content directories from previous runs
    shutil.rmtree('./job_files', ignore_errors=True)
    shutil.rmtree('./training', ignore_errors=True)

    print_json("removed old directories")

    # Ensure only allowed keys are present
    job_input = job['input']
    if 'errors' in (job_input := validate(job_input, INPUT_SCHEMA)):
        return {'error': job_input['errors'], 'print_output': print_output}
    job_input = job_input['validated_input']


    # Download the zip file
    downloaded_input = rp_download.file(job_input['zip_url'])

    if not os.path.exists('./training'):
        os.mkdir('./training')
        os.mkdir('./training/img')
        os.mkdir(
            f"./training/img/{job_input['steps']}_{job_input['instance_name']} {job_input['class_name']}")
        os.mkdir('./training/model')
        os.mkdir('./training/logs')

    # Make clean data directory
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    flat_directory = f"./training/img/{job_input['steps']}_{job_input['instance_name']} {job_input['class_name']}"
    os.makedirs(flat_directory, exist_ok=True)

    print_json("flat_directory: " + flat_directory)

    for root, dirs, files in os.walk(downloaded_input['extracted_path']):
        # Skip __MACOSX folder
        if '__MACOSX' in root:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1].lower() in allowed_extensions:
                shutil.copy(
                    os.path.join(downloaded_input['extracted_path'], file_path),
                    flat_directory
                )

    mc_args = []
    mc_args.append('--batch_size="1"')
    mc_args.append('--num_beams="1"')
    mc_args.append('--top_p="0.9"')
    mc_args.append('--max_length="75"')
    mc_args.append('--min_length="8"')
    mc_args.append('--beam_search')
    mc_args.append('--caption_extension=".txt"')
    mc_args.append('"' + flat_directory + '"')

    make_captions_command = 'python3 ./finetune/make_captions.py ' + ' '.join(mc_args)
    make_captions_command_start = time.time();
    try:
        print_json("make_captions_command: " + make_captions_command)
        subprocess.run(make_captions_command, shell=True, check=True)
    except BaseException as e:
        print_json("make_captions_command failed: " + str(e))

    print_json("make_captions_command ran " + str(time.time() - make_captions_command_start) + " seconds")


    # Input with default values
    output_name = job_input.get('id', "a-traing-specific-name")

    # Accelerate launch arguments
    # See https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch
    # And https://github.com/bmaltais/kohya_ss/wiki/LoRA-training-parameters
    args = []

    num_cpu_threads_per_process = job_input.get('num_cpu_threads_per_process', 0)
    if num_cpu_threads_per_process > 0:
        args.append('--num_cpu_threads_per_process=' + str(num_cpu_threads_per_process))

    # args.append('--num_cpu_threads_per_process=' + str(job_input.get('num_cpu_threads_per_process', 1)))
    args.append('./sdxl_train_network.py')

    args.append('--bucket_no_upscale')
    args.append('--bucket_reso_steps=64')
    args.append('--cache_latents')
    args.append('--keep_tokens="20"')
    args.append('--learning_rate="' + str(job_input['learning_rate']) + '"')
    args.append('--logging_dir="./training/logs"')
    args.append('--lr_scheduler_num_cycles="' + str(job_input['lr_scheduler_num_cycles']) + '"')
    args.append('--lr_scheduler="' + job_input['lr_scheduler'] + '"')
    # args.append('--max_data_loader_n_workers="' + job_input['max_data_loader_n_workers'] + '"')
    args.append('--max_train_steps="' + str(job_input['max_train_steps']) + '"')
    args.append('--mixed_precision="' + job_input['mixed_precision'] + '"')
    args.append('--network_alpha="8"')
    args.append('--network_dim="' + str(job_input['network_dim']) + '"')
    args.append('--network_module=networks.lora')
    args.append('--network_train_unet_only')
    args.append('--no_half_vae')
    args.append('--noise_offset=0.1')
    args.append('--optimizer_type="' + job_input['optimizer_type'] + '"')
    args.append('--output_dir="./training/model"')
    args.append('--output_name="' + output_name + '"')
    args.append('--pretrained_model_name_or_path="/model_cache/sd_xl_base_1.0.safetensors"')
    args.append('--resolution="1024,1024"')
    args.append('--save_every_n_epochs="1"')
    args.append('--save_model_as=safetensors')
    args.append('--save_precision="' + job_input['save_precision'] + '"')
    args.append('--text_encoder_lr=5e-05')
    args.append('--train_batch_size="' + str(job_input['train_batch_size']) + '"')
    args.append('--train_data_dir="./training/img"')
    args.append('--unet_lr="' + str(job_input['unet_lr']) + '"')
    args.append('--xformers')

#     --network_dim=16 --output_name="last" --lr_scheduler_num_cycles="2" --no_half_vae --learning_rate="4e-07" --lr_scheduler="constant" --train_batch_size="1" --max_train_steps="1000" --save_every_n_epochs="1" --mixed_precision="fp16" --save_precision="fp16" --cache_latents --optimizer_type="Adafactor" --max_data_loader_n_workers="0" --keep_tokens="20" --bucket_reso_steps=64 --xformers --bucket_no_upscale --noise_offset=0.1 --network_train_unet_only

        # https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch
    accelerate_launch_command = 'accelerate launch ' + ' '.join(args)
    accelerate_launch_command_start = time.time();
    try:
        print_json("accelerate_launch_command: " + accelerate_launch_command)
        accelerate_launch_command_start = time.time();
        subprocess.run(accelerate_launch_command, shell=True, check=True)
    except BaseException as e:
        print_json("accelerate_launch_command failed: " + str(e))

    print_json("accelerate_launch_command ran " + str(time.time() - accelerate_launch_command_start) + " seconds")

    print_json(list_disk_usage('./training', True))

    uploaded_lora_url = None
    try:
        job_s3_config = job.get('s3Config')

        uploaded_lora_url = upload_file_to_bucket(
            file_name=f"{job['id']}.safetensors",
            file_location=f"./training/model/{output_name}.safetensors",
            bucket_creds=job_s3_config,
            bucket_name=None if job_s3_config is None else job_s3_config['bucketName'],
        )
    except BaseException as e:
        print_json("upload_file_to_bucket failed: " + str(e))

    return {"lora": uploaded_lora_url, "print_json_output": print_json_output}


runpod.serverless.start({"handler": handler})
