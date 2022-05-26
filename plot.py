import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def set_font(font_path='', family=''):
    if family:
        font_name = family
    else:
        font_name = fm.FontProperties(fname=font_path, size=50).get_name()
    plt.rc('font', family=font_name)


def print_font_list():
    font_list = fm.findSystemFonts() 
    font_list.sort()
    fnames = []
    for fpath in font_list:
        #폰트 파일의 경로를 사용하여 폰트 속성 객체 가져오기
        fp=fm.FontProperties(fname=fpath)
        
        # 폰트 속성을 통해 파이썬에 설정해야 하는 폰트 이름 조회 
        font_name=fp.get_name() 
        
        fnames.append(font_name)
        
    for idx, fname in enumerate(fnames):
        print(str(idx).ljust(4), fname)
     
    input()


if __name__ == '__main__':
    import matplotlib as mpl
    print_font_list()

    print(mpl.matplotlib_fname())
    print(mpl.get_cachedir())
    set_font(family='BM JUA_TTF')