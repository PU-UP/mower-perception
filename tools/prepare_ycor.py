"""Validate and freeze the official YCOR data as a traversable-grass proxy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import argparse
import tarfile

import numpy as np
from PIL import Image

PALETTE = [255,255,255, 178,176,153, 128,255,0, 156,76,30, 255,0,128,
           255,0,0, 0,160,0, 40,80,0, 1,88,255]
ARCHIVE_SHA256 = '2a18c82e05aee66bb49480d3a74fcc73ba0129418b7b26a4a7d4a57e67c5cbe2'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_binary(path, size=None):
    with Image.open(path) as image:
        if image.mode != 'P' or image.getpalette()[:27] != PALETTE:
            raise ValueError('Unexpected YCOR indexed PNG palette')
        labels = np.asarray(image)
        if not np.isin(labels, range(9)).all():
            raise ValueError('Unknown YCOR label; do not silently ignore it')
        if size is not None:
            labels = np.asarray(image.resize(size, Image.Resampling.NEAREST))
        result = (labels == 2).astype(np.uint8)
        result[labels == 0] = 255
        return result


def validate_split(manifest):
    if not manifest.get('frozen') or manifest.get('task') != 'ycor_traversable_grass_proxy':
        raise ValueError('Need a frozen YCOR proxy manifest')
    samples = manifest['samples']
    if len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Duplicate sample ID')
    groups, hashes = {}, {}
    for s in samples:
        if s['split'] not in ('train','val','test'):
            raise ValueError('Unknown split')
        for key, index in [('group',groups),('decoded_sha256',hashes),('image_sha256',hashes)]:
            value = s[key]
            if not value:
                raise ValueError('Missing group or digest')
            if value in index and index[value] != s['split']:
                raise ValueError('Cross-split sequence or duplicate leakage')
            index[value] = s['split']
    if {s['split'] for s in samples} != {'train','val','test'}:
        raise ValueError('Every split must be nonempty')


def checked_samples(root, manifest, split):
    validate_split(manifest)
    rows = [s for s in manifest['samples'] if s['split'] == split]
    for s in rows:
        for key in ('image','mask'):
            path = (root/s[key]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Path escapes dataset root')
            if sha256(path) != s[key+'_sha256']:
                raise ValueError('Dataset file digest changed')
        with Image.open(root/s['image']) as image:
            truth = read_binary(root/s['mask'])
            if truth.shape != (image.height,image.width):
                raise ValueError('Annotation alignment mismatch')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/ycor/official-package.tar.gz'))
    parser.add_argument('--root', type=Path, default=Path('data/ycor/yamaha_v0'))
    parser.add_argument('--output', type=Path, default=Path('evaluation/ycor/split.json'))
    args = parser.parse_args()
    if args.output.exists(): parser.error('Frozen manifest exists; do not overwrite it')
    if sha256(args.archive) != ARCHIVE_SHA256:
        raise ValueError('Official archive SHA-256 mismatch; refusing extraction')
    args.root.parent.mkdir(parents=True,exist_ok=True)
    ancillary=[]
    with tarfile.open(args.archive) as archive:
        for member in archive:
            dest=(args.root.parent/member.name).resolve()
            if not dest.is_relative_to(args.root.resolve()) or member.issym() or member.islnk():
                raise ValueError('Unsafe archive member')
            if member.isdir(): dest.mkdir(parents=True,exist_ok=True)
            elif member.isfile():
                dest.parent.mkdir(parents=True,exist_ok=True)
                content=archive.extractfile(member).read()
                dest.write_bytes(content)
                if dest.suffix.lower() not in ('.jpg','.png'):
                    ancillary.append({'path':member.name,'sha256':sha256(dest),'bytes':len(content)})
            else: raise ValueError('Unexpected archive entry')
    samples, hashes, thumbnails=[],[],[]
    for official in ('train','valid'):
        for directory in sorted((args.root/official).glob('iid*')):
            image_path,mask_path=directory/'rgb.jpg',directory/'labels.png'
            number=int(directory.name[3:])
            # Complete source/camera cohort held out, NOT random adjacent frames.
            # Source cohort inferred by visual audit; NOT a published timestamp map.
            split='test' if official=='valid' else ('val' if number>=1053 else 'train')
            group={'train':'official-train-white-vehicle-cohort',
                   'val':'official-train-red-vehicle-cohort',
                   'test':'official-valid-session-holdout'}[split]
            with Image.open(image_path) as source:
                rgb=source.convert('RGB')
                truth=read_binary(mask_path)
                if truth.shape != (rgb.height,rgb.width): raise ValueError('Image/label geometry mismatch')
                decoded=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
                a=np.asarray(rgb.convert('L').resize((9,8),Image.Resampling.BILINEAR))
                hashes.append(int.from_bytes(np.packbits(a[:,1:]>a[:,:-1]).tobytes(),'big'))
                thumbnails.append(np.asarray(rgb.resize((32,16),Image.Resampling.BILINEAR),dtype=np.float32))
                size=list(rgb.size)
            valid=truth!=255
            samples.append({'id':directory.name,'official_split':official,'split':split,'group':group,
                 'report_group':f'official-valid-iid-{(number-839)//50}' if official=='valid' else group,
                 'image':str(image_path.relative_to(args.root)),'mask':str(mask_path.relative_to(args.root)),
                 'image_sha256':sha256(image_path),'mask_sha256':sha256(mask_path),'decoded_sha256':decoded,
                 'size_wh':size,'positive_fraction':float((truth==1).sum()/valid.sum()),
                 'ignored_pixels':int((~valid).sum())})
    assert sum(s['official_split']=='train' for s in samples)==931
    assert sum(s['official_split']=='valid' for s in samples)==145
    # Conservative filtering uses only image similarity, never model outcomes.
    candidates,excluded=[],set()
    priority={'train':0,'val':1,'test':2}
    for i,a in enumerate(samples):
        for j in range(i+1,len(samples)):
            b=samples[j]
            if a['split']==b['split']:continue
            hamming=(hashes[i]^hashes[j]).bit_count()
            exact=a['decoded_sha256']==b['decoded_sha256'] or a['image_sha256']==b['image_sha256']
            if hamming<=6 or exact:
                rmse=float(np.sqrt(np.mean((thumbnails[i]-thumbnails[j])**2)))
                if exact or rmse<=18:
                    remove=a if priority[a['split']]<priority[b['split']] else b
                    excluded.add(remove['id'])
                    candidates.append({'a':a['id'],'b':b['id'],'hamming64':hamming,'rmse32x16':rmse,
                                       'exact':exact,'excluded':remove['id']})
    manifest={'task':'ycor_traversable_grass_proxy','frozen':True,'seed':20260921,
       'source':'https://theairlab.org/yamaha-offroad-dataset/',
       'download':'https://cmu.box.com/shared/static/3fngoljhcwhqf2z5cbepufh331qtesxt.gz',
       'archive_sha256':sha256(args.archive),'license':'CC BY 4.0 per official landing page',
       'citation':'Maturana, Chou, Uenoyama, Scherer (2018), Real-time semantic mapping for autonomous off-road navigation, Field and Service Robotics, 335-350',
       'grouping':'Official valid retained as test; author states train/valid session-disjoint but location-overlapping. Internal validation uses the complete visually distinct red-vehicle source cohort (official train iid >=1053). Internal source grouping is inferred, NOT proven by timestamps. No random per-frame split.',
       'report_groups':'Five 50-ID diagnostic bins within official valid; NOT independent collection sessions or locations. Split leakage uses group, not these bins.',
       'label_mapping':{'0':'ignore (unannotated white)','1':'other: smooth trail','2':'positive: traversable grass',
                        '3':'other: rough trail','4':'other: puddle','5':'other: obstacle','6':'other: non-traversable low vegetation',
                        '7':'other: high vegetation','8':'other: sky'},
       'ancillary_files':ancillary,'near_duplicate_rule':'cross-split exact decoded/encoded digest OR (64-bit dHash Hamming <=6 AND RGB 32x16 RMSE <=18); preserve test, then val, exclude lower-priority sample',
       'near_duplicate_limitations':'heuristic misses viewpoint/illumination changes; does not prove session independence',
       'cross_split_candidates':candidates,'excluded_ids':sorted(excluded),
       'samples':[s for s in samples if s['id'] not in excluded]}
    validate_split(manifest)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:f.write(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'counts':{k:sum(s['split']==k for s in manifest['samples']) for k in ('train','val','test')},
                      'excluded':len(excluded),'ancillary':ancillary,'manifest_sha256':sha256(args.output)},indent=2))


if __name__=='__main__':main()
