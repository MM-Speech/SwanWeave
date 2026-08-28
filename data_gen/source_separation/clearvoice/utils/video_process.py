import torch

import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, pdb, math, python_speech_features
import numpy as np
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score
import soundfile as sf

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from ..models.av_mossformer2_tse.faceDetector.s3fd import S3FD

from .decode import decode_one_audio_AV_MossFormer2_TSE_16K



def process_tse(args, model, device, data_reader, output_wave_dir):
	video_args = args_param()
	video_args.model = model
	video_args.device = device
	video_args.sampling_rate = args.sampling_rate
	args.device = device
	assert args.sampling_rate == 16000
	with torch.no_grad():
		for videoPath in data_reader:  # Loop over all video samples
			savFolder = videoPath.split(os.path.sep)[-1]
			video_args.savePath = f'{output_wave_dir}/{savFolder.split(".")[0]}/'
			video_args.videoPath = videoPath
			main(video_args, args)



def args_param():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
    parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
    parser.add_argument('--minTrack',              type=int,   default=50,   help='Number of min frames for each shot')
    parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
    parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
    parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')
    parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
    parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')
    video_args = parser.parse_args()
    return video_args


# Main function
def main(video_args, args):
    # Initialization 
    video_args.pyaviPath = os.path.join(video_args.savePath, 'py_video')
    video_args.pyframesPath = os.path.join(video_args.savePath, 'pyframes')
    video_args.pyworkPath = os.path.join(video_args.savePath, 'pywork')
    video_args.pycropPath = os.path.join(video_args.savePath, 'py_faceTracks')
    if os.path.exists(video_args.savePath):
        rmtree(video_args.savePath)
    os.makedirs(video_args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
    os.makedirs(video_args.pyframesPath, exist_ok = True) # Save all the video frames
    os.makedirs(video_args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
    os.makedirs(video_args.pycropPath, exist_ok = True) # Save the detected face clips (audio+video) in this process

    # Extract video
    video_args.videoFilePath = os.path.join(video_args.pyaviPath, 'video.avi')
    # If duration did not set, extract the whole video, otherwise extract the video from 'video_args.start' to 'video_args.start + video_args.duration'
    if video_args.duration == 0:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % \
            (video_args.videoPath, video_args.nDataLoaderThread, video_args.videoFilePath))
    else:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % \
            (video_args.videoPath, video_args.nDataLoaderThread, video_args.start, video_args.start + video_args.duration, video_args.videoFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(video_args.videoFilePath))

    # Extract audio
    video_args.audioFilePath = os.path.join(video_args.pyaviPath, 'audio.wav')
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
        (video_args.videoFilePath, video_args.nDataLoaderThread, video_args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(video_args.audioFilePath))

    # Extract the video frames
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
        (video_args.videoFilePath, video_args.nDataLoaderThread, os.path.join(video_args.pyframesPath, '%06d.jpg'))) 
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(video_args.pyframesPath))

    # Scene detection for the video frames
    scene = scene_detect(video_args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(video_args.pyworkPath))	

    # Face detection for the video frames
    faces = inference_video(video_args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(video_args.pyworkPath))

    # Face tracking
    allTracks, vidTracks = [], []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= video_args.minTrack: # Discard the shot frames less than minTrack frames
            allTracks.extend(track_shot(video_args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))

    # Face clips cropping
    for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
        vidTracks.append(crop_video(video_args, track, os.path.join(video_args.pycropPath, '%05d'%ii)))
    savePath = os.path.join(video_args.pyworkPath, 'tracks.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(vidTracks, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %video_args.pycropPath)
    fil = open(savePath, 'rb')
    vidTracks = pickle.load(fil)
    fil.close()

    # AVSE
    files = glob.glob("%s/*.avi"%video_args.pycropPath)
    files.sort()

    est_sources = evaluate_network(files, video_args, args)

    visualization(vidTracks, est_sources, video_args)	

    # combine files in pycrop
    for idx, file in enumerate(files):
        print(file)
        command = f"ffmpeg -i {file} {file[:-9]}orig_{idx}.mp4 ;"
        command += f"rm {file} ;"
        command += f"rm {file.replace('.avi', '.wav')} ;"

        command += f"ffmpeg -i {file[:-9]}orig_{idx}.mp4 -i {file[:-9]}est_{idx}.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest {file[:-9]}est_{idx}.mp4 ;"
        # command += f"rm {file[:-9]}est_{idx}.wav ;"

        output = subprocess.call(command, shell=True, stdout=None)

    rmtree(video_args.pyworkPath)
    rmtree(video_args.pyframesPath)




def scene_detect(video_args):
	# CPU: Scene detection, output is the list of each shot's time duration
	videoManager = VideoManager([video_args.videoFilePath])
	statsManager = StatsManager()
	sceneManager = SceneManager(statsManager)
	sceneManager.add_detector(ContentDetector())
	baseTimecode = videoManager.get_base_timecode()
	videoManager.set_downscale_factor()
	videoManager.start()
	sceneManager.detect_scenes(frame_source = videoManager)
	sceneList = sceneManager.get_scene_list(baseTimecode)
	savePath = os.path.join(video_args.pyworkPath, 'scene.pckl')
	if sceneList == []:
		sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
	with open(savePath, 'wb') as fil:
		pickle.dump(sceneList, fil)
		sys.stderr.write('%s - scenes detected %d\n'%(video_args.videoFilePath, len(sceneList)))
	return sceneList

def inference_video(video_args):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(
		device=video_args.device,
		model_root=getattr(video_args, 'model_root', None),
		allow_legacy_fallback=getattr(video_args, 'allow_legacy_fallback', True),
	)
	flist = glob.glob(os.path.join(video_args.pyframesPath, '*.jpg'))
	flist.sort()
	dets = []
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[video_args.facedetScale])
		dets.append([])
		for bbox in bboxes:
		  dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
		sys.stderr.write('%s-%05d; %d dets\r' % (video_args.videoFilePath, fidx, len(dets[-1])))
	savePath = os.path.join(video_args.pyworkPath,'faces.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(dets, fil)
	return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def track_shot(video_args, sceneFaces):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= video_args.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > video_args.minTrack:
			frameNum    = np.array([ f['frame'] for f in track ])
			bboxes      = np.array([np.array(f['bbox']) for f in track])
			frameI      = np.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = np.stack(bboxesI, axis=1)
			if max(np.mean(bboxesI[:,2]-bboxesI[:,0]), np.mean(bboxesI[:,3]-bboxesI[:,1])) > video_args.minFaceSize:
				tracks.append({'frame':frameI,'bbox':bboxesI})
	return tracks

def crop_video(video_args, track, cropFile):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(video_args.pyframesPath, '*.jpg')) # Read the frames
	flist.sort()
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
		dets['y'].append((det[1]+det[3])/2) # crop center x 
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	for fidx, frame in enumerate(track['frame']):
		cs  = video_args.cropScale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
		image = cv2.imread(flist[frame])
		frame = np.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		vOut.write(cv2.resize(face, (224, 224)))
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (video_args.audioFilePath, video_args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, video_args.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets}


def evaluate_network(files, video_args, args):

	est_sources = []
	for file in tqdm.tqdm(files, total = len(files)):

		fileName = os.path.splitext(file.split(os.path.sep)[-1])[0] # Load audio and video
		audio, _ = sf.read(os.path.join(video_args.pycropPath, fileName + '.wav'), dtype='float32')

		video = cv2.VideoCapture(os.path.join(video_args.pycropPath, fileName + '.avi'))
		videoFeature = []
		while video.isOpened():
			ret, frames = video.read()
			if ret == True:
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224,224))
				face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
				videoFeature.append(face)
			else:
				break

		video.release()
		visual = np.array(videoFeature)/255.0
		visual = (visual - 0.4161)/0.1688

		length = int(audio.shape[0]/16000*25)
		if visual.shape[0] < length:
			visual = np.pad(visual, ((0,int(length - visual.shape[0])),(0,0),(0,0)), mode = 'edge')

		audio /= np.max(np.abs(audio))
		audio = np.expand_dims(audio, axis=0)
		visual = np.expand_dims(visual, axis=0)

		inputs = (audio, visual)
		est_source = decode_one_audio_AV_MossFormer2_TSE_16K(video_args.model, inputs, args)

		est_sources.append(est_source)

	return est_sources

def visualization(tracks, est_sources, video_args):
	# CPU: visulize the result for video format
	flist = glob.glob(os.path.join(video_args.pyframesPath, '*.jpg'))
	flist.sort()
	

	for idx, audio in enumerate(est_sources):
		max_value = np.max(np.abs(audio))
		if max_value >1:
			audio /= max_value
		sf.write(video_args.pycropPath +'/est_%s.wav' %idx, audio, 16000)

	for tidx, track in enumerate(tracks):
		faces = [[] for i in range(len(flist))]
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			faces[frame].append({'track':tidx, 's':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	
		firstImage = cv2.imread(flist[0])
		fw = firstImage.shape[1]
		fh = firstImage.shape[0]
		vOut = cv2.VideoWriter(os.path.join(video_args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
		for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
			image = cv2.imread(fname)
			for face in faces[fidx]:
				cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,255,0),10)
			vOut.write(image)
		vOut.release()

		command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
			(os.path.join(video_args.pyaviPath, 'video_only.avi'), (video_args.pycropPath +'/est_%s.wav' %tidx), \
			video_args.nDataLoaderThread, os.path.join(video_args.pyaviPath,'video_out_%s.avi'%tidx))) 
		output = subprocess.call(command, shell=True, stdout=None)




		command = "ffmpeg -i %s %s ;" % (
					os.path.join(video_args.pyaviPath, 'video_out_%s.avi' % tidx),
					os.path.join(video_args.pyaviPath, 'video_est_%s.mp4' % tidx)
				)
		command += f"rm {os.path.join(video_args.pyaviPath, 'video_out_%s.avi' % tidx)}"
		output = subprocess.call(command, shell=True, stdout=None)


	command = "ffmpeg -i %s %s ;" % (
				os.path.join(video_args.pyaviPath, 'video.avi'),
				os.path.join(video_args.pyaviPath, 'video_orig.mp4')
			)
	command += f"rm {os.path.join(video_args.pyaviPath, 'video_only.avi')} ;"
	command += f"rm {os.path.join(video_args.pyaviPath, 'video.avi')} ;"
	command += f"rm {os.path.join(video_args.pyaviPath, 'audio.wav')} ;"
	output = subprocess.call(command, shell=True, stdout=None)
