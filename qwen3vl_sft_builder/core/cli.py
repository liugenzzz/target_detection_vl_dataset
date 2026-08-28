"""脚本入口的统一包装。

配置类错误（YAML 写错、路径没设、模型服务 401）是用户改配置就能解决的，
甩一屏 Python 堆栈只会把那句有用的提示淹掉。这里把它们翻译成人话。
真正的程序 bug 仍然照常抛出，堆栈该看还得看。

原先四个脚本各抄了一份一模一样的实现。
"""

from __future__ import annotations

from .vlm_client import FatalVlmError


def _cli(entry):
    """配置类错误直接打印人话并退出，不甩 Python 堆栈 —— 这类错误是用户改配置就能解决的，
    堆栈只会淹没真正有用的那句提示。真正的程序 bug 仍然照常抛出。"""
    import sys
    try:
        return entry()
    except FatalVlmError as exc:
        print(f"\n模型服务配置有问题，已中止（重试也不会成功）：\n\n    {exc}\n\n"
              f"改好后先跑 python scripts/check_vlm.py 确认三步都通过。\n",
              file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError) as exc:
        print(f"\n配置有问题：\n{exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断。已完成的 VLM 结果都在缓存里，重跑会从断点继续。", file=sys.stderr)
        return 130
