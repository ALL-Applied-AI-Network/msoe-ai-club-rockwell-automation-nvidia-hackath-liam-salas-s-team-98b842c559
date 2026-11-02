"""Deterministic epoch learning-rate schedule, without optimizer state resets."""
import math

def epoch_learning_rate(protocol, epoch):
    """Return the fixed protocol rate for one nominal training epoch."""
    config=protocol['optimizer'];schedule=config['schedule'];epochs=protocol['epochs']
    if type(epoch) is not int or type(epochs) is not int or epochs<2 or not 0<=epoch<epochs:
        raise ValueError('Invalid nominal epoch')
    if config['name']!='AdamW' or schedule['name']!='epoch_cosine' or schedule['nominal_epochs']!=epochs:
        raise ValueError('Unsupported schedule contract')
    base=config['learning_rate'];minimum=schedule['minimum_learning_rate']
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (base,minimum)) or not 0<minimum<=base:
        raise ValueError('Invalid learning-rate bounds')
    if epoch==0:return float(base)
    if epoch==epochs-1:return float(minimum)
    return minimum+(base-minimum)*(1+math.cos(math.pi*epoch/(epochs-1)))/2

def set_epoch_learning_rate(optimizer, protocol, epoch):
    """Apply the protocol rate to all groups while retaining optimizer state."""
    value=epoch_learning_rate(protocol,epoch)
    if not optimizer.param_groups:raise ValueError('Missing optimizer groups')
    for group in optimizer.param_groups:group['lr']=value
    return value
