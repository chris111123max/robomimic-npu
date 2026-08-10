import argparse
import os
from pathlib import Path
from copy import deepcopy

import imageio
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


def rollout(policy, env, horizon, video_writer, video_skip, camera_names):
    policy.start_episode()

    obs = env.reset()
    state_dict = env.get_state()

    # robosuite deterministic reset
    obs = env.reset_to(state_dict)

    total_reward = 0.0
    success = False
    video_count = 0

    for step_i in range(horizon):

        # 1. policy根据当前obs输出动作
        act = policy(ob=obs)

        # 2. 环境执行动作
        next_obs, reward, done, _ = env.step(act)

        total_reward += reward

        # 3. 判断任务是否成功
        success = env.is_success()["task"]

        # 4. 录像
        if video_count % video_skip == 0:
            imgs = []

            for cam_name in camera_names:
                img = env.render(
                    mode="rgb_array",
                    height=512,
                    width=512,
                    camera_name=cam_name,
                )
                imgs.append(img)

            frame = np.concatenate(imgs, axis=1)
            video_writer.append_data(frame)

        video_count += 1

        # 成功或者环境结束就停止
        if done or success:
            break

        obs = deepcopy(next_obs)

    return {
        "Return": total_reward,
        "Horizon": step_i + 1,
        "Success": bool(success),
    }


def main(args):

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # NPU / device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    print("device =", device)

    # 加载训练好的policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent,
        device=device,
        verbose=True,
    )

    # horizon
    if args.horizon is None:
        config, _ = FileUtils.config_from_checkpoint(
            ckpt_dict=ckpt_dict
        )
        horizon = config.experiment.rollout.horizon
    else:
        horizon = args.horizon

    # 创建环境
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=None,
        render=False,
        render_offscreen=True,
        verbose=True,
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_success = 0
    results = []

    print()
    print("=" * 60)
    print("Start evaluation")
    print("Rollouts :", args.n_rollouts)
    print("Horizon  :", horizon)
    print("Seed     :", args.seed)
    print("Camera   :", args.camera_names)
    print("=" * 60)

    for ep in range(1, args.n_rollouts + 1):

        # 每一局先写临时视频
        temp_video = output_dir / f".temp_episode_{ep:02d}.mp4"

        writer = imageio.get_writer(
            str(temp_video),
            fps=20,
        )

        try:
            stats = rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                video_writer=writer,
                video_skip=args.video_skip,
                camera_names=args.camera_names,
            )
        finally:
            writer.close()

        results.append(stats)

        if stats["Success"]:

            num_success += 1

            final_video = output_dir / (
                f"success_{num_success:02d}"
                f"_episode_{ep:02d}"
                f"_horizon_{stats['Horizon']}.mp4"
            )

            os.replace(temp_video, final_video)

            print(
                f"[Episode {ep:02d}] SUCCESS | "
                f"Horizon={stats['Horizon']} | "
                f"Return={stats['Return']:.3f} | "
                f"Saved={final_video}"
            )

        else:

            # 失败录像直接删除
            if temp_video.exists():
                temp_video.unlink()

            print(
                f"[Episode {ep:02d}] FAIL    | "
                f"Horizon={stats['Horizon']} | "
                f"Return={stats['Return']:.3f} | "
                f"video deleted"
            )

    success_rate = num_success / args.n_rollouts

    print()
    print("=" * 60)
    print("Evaluation Finished")
    print(f"Num Rollouts  : {args.n_rollouts}")
    print(f"Num Success   : {num_success}")
    print(f"Success Rate  : {success_rate:.3f}")
    print(f"Saved Videos  : {num_success}")
    print(f"Output Dir    : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--video_skip",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--camera_names",
        nargs="+",
        default=["frontview"],
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
    )

    args = parser.parse_args()
    main(args)
