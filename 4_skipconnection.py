import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_skip=True):
        super(ResidualBlock, self).__init__()

        # 기본 합성곱 블록 (Conv → BN → ReLU → Conv → BN)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # ✅ Skip Connection이 가능하도록 입력을 변환하는 레이어 (차원이 다를 경우만 적용)
        self.skip_connection = None
        if use_skip and (in_channels != out_channels or stride != 1):  # 차원이 다르면 적용
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x  # 원본 입력 (Skip Connection에 사용)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # ✅ Skip Connection 적용 (입력과 출력의 크기가 다르면 변환)
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)  # 입력을 변환

        out += identity  # Skip Connection 수행
        out = self.relu(out)  # 최종 활성화 함수

        return out