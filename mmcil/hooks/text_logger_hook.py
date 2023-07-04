from mmcv.runner import TextLoggerHook
from mmcv.runner.hooks import HOOKS


@HOOKS.register_module()
class TextLoggerHookCIL(TextLoggerHook):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.has_start = False

    def before_train_epoch(self, runner) -> None:
        if not self.has_start:
            self.has_start = True
            super(TextLoggerHookCIL, self).after_train_epoch(runner)
        else:
            super(TextLoggerHookCIL, self).before_train_epoch(runner)
