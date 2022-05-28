import torch
import matplotlib.pyplot as plt 
import random
import numpy as np

from rich import print


def separate_by_space(x, kernel_width=1, min_brightness=3, min_space=50, min_letter_len=5, 
    result=None, index=None, progress=None, task_id=None,
):
    '''
    Parameters
    ----------
        ``kernel_width`` ``(int)`` : 
            글자를 감지할 필터의 너비
        ``min_brightness`` ``(float)`` : 
            글자의 감지 여부를 판단할 최소 필터 밝기
            필터 밝기: 세로 한줄의 0~1 범위 픽셀을 ``sum()`` 한 결과
        ``min_space`` ``(int)`` : 
            띄어쓰기로 판단할 최소 빈칸 길이
        ``min_letter_len`` ``(int)`` :
            글자의 감지 여부를 판단할 최소 글자 너비
    '''
    
    row_len = x.shape[2] # (C, H, (W))
    col_len = x.shape[1] # (C, (H), W)

    sep_idxs = []
    brightness_list = []
    detected = False
    appended = False
    space = 0
    xl = 0
    xr = 0

    for now_x in range(1, row_len - kernel_width + 1, 1):
        v_line = x[:, :, now_x : now_x+kernel_width ]
        brightness = torch.sum(v_line)
        brightness_list.append(brightness)

        if brightness > min_brightness: # 글씨가 감지되면
            if not detected and appended: # 새로운 감지가 시작되는 순간
                appended = False
                xl = now_x
            space = 0
            detected = True

        else: # 글씨가 감지되지 않으면
            space += 1
            if detected: # 감지가 종료되는 순간
                xr = now_x
            if space == min_space: # 공백 수가 띄어쓰기 너비를 달성하면
                if xr-xl > min_letter_len: 
                    sep_idxs.append((0, col_len, xl, xr))
                appended = True
            detected = False

    if detected and not appended and xr-xl > min_letter_len: # 마지막 감지가 종료되지 않았다면
        sep_idxs.append((0, col_len, xl, row_len))
    
    # 얻어낸 데이터 처리
    if isinstance(result, list) and index is not None:
        result[index] = (*sep_idxs,)
    else:
        return (*sep_idxs,), brightness_list
    
    if progress is not None:
        progress.update(task_id, advance=1)


def save_n_piece(x, kernel_width=1, min_brightness=3, min_space=50, min_letter_len=5, 
    result=None, index=None, progress=None, task_id=None,
):
    '''
    Parameters
    ----------
        ``kernel_width`` ``(int)`` : 
            글자를 감지할 필터의 너비
        ``min_brightness`` ``(float)`` : 
            글자의 감지 여부를 판단할 최소 필터 밝기
            필터 밝기: 세로 한줄의 0~1 범위 픽셀을 ``sum()`` 한 결과
        ``min_space`` ``(int)`` : 
            띄어쓰기로 판단할 최소 빈칸 길이
        ``min_letter_len`` ``(int)`` :
            글자의 감지 여부를 판단할 최소 글자 너비
    '''
    
    row_len = x.shape[2] # (C, H, (W))

    n_sep_idxs = 0
    detected = False
    appended = False
    space = 0
    xl = 0
    xr = 0

    for now_x in range(1, row_len - kernel_width + 1, 1):
        kernel = x[:, :, now_x : now_x+kernel_width ]
        brightness = torch.sum(kernel)

        if brightness > min_brightness: # 글씨가 감지되면
            if not detected and appended: # 새로운 감지가 시작되는 순간
                appended = False
                xl = now_x
            space = 0
            detected = True

        else: # 글씨가 감지되지 않으면
            space += 1
            if detected: # 감지가 종료되는 순간
                xr = now_x
            if space == min_space: # 공백 수가 띄어쓰기 너비를 달성하면
                if xr-xl > min_letter_len: 
                    n_sep_idxs += 1
                appended = True
            detected = False

    if detected and not appended and xr-xl > min_letter_len: # 마지막 감지가 종료되지 않았다면
        n_sep_idxs += 1    

    # 얻어낸 데이터 처리
    if isinstance(result, list) and index is not None:
        result[index] = n_sep_idxs
    else:
        return n_sep_idxs
    
    if progress is not None:
        progress.update(task_id, advance=1)


def crop_blank_piece(x_piece, sep_idx, min_brightness=3):
    base_yt, _, base_xl, _ = sep_idx
    row_len = x_piece.shape[2] # (C, H, (W))
    col_len = x_piece.shape[1] # (C, (H), W)

    yt = 0
    while torch.sum(x_piece[:, yt:yt+1, :]) < min_brightness and yt < col_len:
        yt += 1

    yb = col_len
    while torch.sum(x_piece[:, yb:yb+1, :]) < min_brightness and yb >= 0:
        yb -= 1

    xl = 0
    while torch.sum(x_piece[:, :, xl:xl+1]) < min_brightness and xl < row_len:
        xl += 1

    xr = row_len
    while torch.sum(x_piece[:, :, xr:xr+1]) < min_brightness and xr >= 0:
        xr -= 1

    return (base_yt + yt, base_yt + yb, base_xl + xl, base_xl + xr)


def crop_blank(x, sep_idxs):
    croping_idxs = []
    for sep_idx in sep_idxs:
        yt, yb, xl, xr = sep_idx
        x_piece = x[:, yt:yb, xl:xr]
        croping_idx = crop_blank_piece(x_piece, sep_idx)
        croping_idxs.append(croping_idx)
    
    return croping_idxs


def get_data_from_train_set(train_set, progress, func, random_choice=False, **kwargs):
    result = [None] * len(train_set)

    task_id = progress.add_task(f"[yellow]{func.__name__}", total=len(train_set))

    for idx, (x, _) in enumerate(train_set):
        if random_choice: 
            x, _ = random.choice(train_set)
        kwargs={
            'x': x,
            'result': result,
            'index': idx,
            'progress': progress,
            'task_id': task_id,
            **kwargs,
        }
        func(**kwargs)

    return result


def get_corrct_rate_n_piece(train_set, n_pieces_set, len_str=True):
    n_corrct_n_piece = 0
    
    for (_, t), y_n_piece in zip(train_set, n_pieces_set):
        t = t['text']
        t_n_piece = len(t.split()) if len_str else len(t)
        if y_n_piece == t_n_piece:
            n_corrct_n_piece += 1

    return n_corrct_n_piece / len(train_set) * 100
   