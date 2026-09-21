from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from mowerseg.ycor import proxy_counts, proxy_metrics, load_checkpoint, ProxyPredictor


def test_ignore_never_counts_as_negative_or_creates_boundaries():
    truth = np.zeros((20, 30), dtype=np.uint8)
    truth[:, 15:] = 255
    pred = np.zeros_like(truth)
    pred[:, 15:] = 1
    c = proxy_counts(pred, truth)
    assert c['tn'] == 300 and c['ignored'] == 300
    assert c['fp'] == c['fn'] == c['tp'] == 0
    assert c['true_boundary'] == c['pred_boundary'] == 0
    assert proxy_metrics(c)['grass_false_negative_rate'] is None
    truth[:, 7:12] = 1
    pred[:, 7:12] = 1
    assert proxy_metrics(proxy_counts(pred, truth))['grass_iou'] == 1
    assert proxy_metrics(proxy_counts(pred, truth))['boundary_f1'] == 1


def test_proxy_formulas_and_invalid_values():
    c = proxy_counts(np.array([[1,1,0,0,1]]), np.array([[1,0,1,0,255]]),0)
    m = proxy_metrics(c)
    assert m['grass_iou'] == 1/3
    assert m['grass_false_positive_rate'] == m['grass_false_negative_rate'] == .5
    assert m['predicted_grass_error_fraction'] == .5
    with pytest.raises(ValueError): proxy_counts(np.zeros((2,2)),np.full((2,2),2))


def test_ycor_mapping_and_group_leakage(tmp_path):
    from tools.prepare_ycor import read_binary, PALETTE, validate_split
    p=tmp_path/'label.png'
    image=Image.fromarray(np.arange(9,dtype=np.uint8)[None,:],mode='P')
    image.putpalette(PALETTE+[0]*(768-len(PALETTE)))
    image.save(p)
    assert read_binary(p).tolist()==[[255,0,1,0,0,0,0,0,0]]
    assert set(np.unique(read_binary(p,(27,3))))=={0,1,255}
    samples=[{'id':str(i),'split':split,'group':str(i),'image_sha256':str(i),
              'decoded_sha256':'decoded'+str(i)} for i,split in enumerate(('train','val','test'))]
    manifest={'frozen':True,'task':'ycor_traversable_grass_proxy','samples':samples}
    validate_split(manifest)
    samples[1]['group']='0'
    with pytest.raises(ValueError,match='leakage'):validate_split(manifest)
    samples[1]['group']='1'; samples[2]['decoded_sha256']='decoded0'
    with pytest.raises(ValueError,match='leakage'):validate_split(manifest)


def test_checkpoint_rejects_untrained_and_wrong_semantics(tmp_path):
    p = tmp_path/'bad.pt'
    torch.save({'task':'ycor_traversable_grass_proxy','optimizer_steps':0},p)
    with pytest.raises(ValueError,match='trained'): load_checkpoint(p,'cpu')
    torch.save({'task':'ycor_traversable_grass_proxy','optimizer_steps':1,'labels':{'0':'ignore'}},p)
    with pytest.raises(ValueError,match='mapping'): load_checkpoint(p,'cpu')


def test_augmentation_keeps_image_and_labels_aligned(monkeypatch):
    from tools.train_ycor import batch, MEAN, STD
    import tools.train_ycor as training
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    monkeypatch.setattr(training.random,'random',lambda:0.)
    monkeypatch.setattr(training.random,'uniform',lambda a,b:1.)
    images=np.zeros((1,2,4,3),np.uint8); images[:,:,:2,0]=255
    labels=np.array([[[1,1,0,255],[1,1,0,255]]],np.uint8)
    x,y=batch(images,labels,np.array([0]),True)
    restored=(x*STD+MEAN)[0,0].numpy()
    np.testing.assert_array_equal(y.numpy(),labels[:,:,::-1])
    np.testing.assert_allclose(restored,[[0,0,1,1],[0,0,1,1]],atol=1e-6)


def test_pipeline_timing_contains_transfers(monkeypatch):
    import mowerseg.ycor as module
    events=[]
    def clock():
        events.append('clock')
        return float(len(events))
    original_to, original_cpu = torch.Tensor.to, torch.Tensor.cpu
    def tracked_to(self,*a,**kw):
        events.append('to')
        return original_to(self,*a,**kw)
    def tracked_cpu(self,*a,**kw):
        events.append('cpu')
        return original_cpu(self,*a,**kw)
    monkeypatch.setattr(module.time,'perf_counter',clock)
    monkeypatch.setattr(torch.Tensor,'to',tracked_to)
    monkeypatch.setattr(torch.Tensor,'cpu',tracked_cpu)
    predictor=ProxyPredictor.__new__(ProxyPredictor)
    predictor.device=torch.device('cpu'); predictor.name='lraspp'; predictor.processor=None
    def model(x):
        events.append('model')
        return {'out':torch.zeros((1,2,8,8))}
    predictor.model=model
    pred,t= predictor.predict(Image.new('RGB',(60,40)))
    clocks=[i for i,e in enumerate(events) if e=='clock']
    assert clocks[0] < events.index('to') < clocks[1]
    assert clocks[1] < events.index('model') < clocks[2]
    assert clocks[2] < events.index('cpu') < clocks[3]
    assert pred.shape==(384,512) and not pred.any()
    assert t['pipeline_ms']==t['preprocess_ms']+t['forward_ms']+t['postprocess_ms']
