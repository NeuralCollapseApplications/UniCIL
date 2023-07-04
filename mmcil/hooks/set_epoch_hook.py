from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class SetEpochHook(Hook):

    def before_train_epoch(self, runner):
        runner.model.module.head.set_epoch((runner.epoch + 1) / runner.max_epochs)
