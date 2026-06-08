class WindowSplitter:
    def __init__(self, window_size: int, stride: int, window_mode: str):
        self.window_size = window_size
        self.stride = stride
        self.window_mode = window_mode
    def split(self, data, time_column=None):
        windows = []
        total_samples = len(data)
        if self.window_mode == 'sliding':
            start = 0
            while start + self.window_size <= total_samples:
                end = start + self.window_size
                windows.append(data.iloc[start:end])
                start += self.stride
        elif self.window_mode == 'fixed':
            for i in range(0, total_samples, self.window_size):
                window = data.iloc[i:i+self.window_size]
                if len(window) == self.window_size:
                    windows.append(window)
        else:
            raise ValueError(f"Unknown window mode: {self.window_mode}")
        return windows
