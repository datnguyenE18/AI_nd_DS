import random

def test():
    """
        0 --- 1 --- 2
        |  \  |  /  |
        3 --- 4 --- 5
      / |  /  |  \  |
     |  6 --- 7 --- 8
      \  --- /

    9 là nút giả cho vị trí thoát, 9 nối 0, 1, 2, 3, 5, 6, 7, 8
    """
    district = [
        [1, 3, 4, 9],#0
        [0, 2, 4, 9],#1
        [1, 4, 5, 9],#2
        [0, 4, 6, 7, 9],#3
        [0, 1, 2, 3, 5, 6, 7, 8],#4
        [2, 4, 8, 9],#5
        [3, 4, 7, 9],#6
        [4, 6, 8, 3, 9],#7
        [4, 5, 7, 9],#8
        [0, 1, 2, 3, 5, 6, 7, 8],#9
    ]
    restaurants = [1, 3, 6, 8]

    # """
    #     0 --- 1
    #     |  \  |
    #     2 --- 3
    #
    # 4 là nút giả cho vị trí thoát, 4 nối 0, 1, 2, 3
    # """
    # # mô tả quận dưới dạng một đồ thị
    # district = [
    #     [1, 2, 3, 4],#0
    #     [0, 3, 4],#1
    #     [0, 3, 4],#2
    #     [0, 1, 2, 4],#3
    #     [0, 1, 2, 3]#4
    # ]
    # # Vị trí các quán nhậu: index của các đỉnh (hoặc là các đỉnh)
    # restaurants = [2, 3]
    # số đỉnh (hoặc số giao điểm các con đường), có tính cả đỉnh giả
    nCrossroads = len(district)-1
    # đỉnh giả (nơi xét là ra khỏi quận)
    endPoint = nCrossroads

    # biến đầu ra: lưu tỉ lệ bắt được người say rượu tại các đỉnh (nếu giả sử có đặt chốt tại đó)
    # catchRates: có kích thước bằng số đỉnh thật
    catchRates = []

    # tỉ lệ lấy mẫu tại mỗi quán nhậu
    nSamplings = 10000

    # xét mỗi vị trí đặt chốt
    # trong vòng lặp thì pol là vị trí đặt chốt đang xét
    for pol in range(nCrossroads):
        #print("pol {}".format(pol))

        # Số lượng bắt được người say rượu tại chốt pol
        nCatches = 0
        # Duyệt từng nhà hàng để thực hiện sinh mẫu người say rượu
        for res in restaurants:
            for i in range(nSamplings):
                # vị trí hiện tại của người say rượu, ban đầu là vị trí quán nhậu
                idx = res
                # tuyến đường đi của người say rượu, ban đầu chỉ có 1 vị trí là quán nhậu
                path = [idx]

                # lặp trong khi chưa thoát hoặc chưa bị bắt
                while idx!=endPoint and idx !=pol:
                    # lấy danh sách các lựa chọn tại mỗi giao điểm
                    choices = district[idx]
                    # Chọn ngẫu nhiên một hướng đi
                    choice = random.randint(0, len(choices)-1)
                    # gán vị trí hiện tại mới và cập nhật vào tuyến đường
                    idx = choices[choice]
                    path.append(idx)
                # nếu bị bắt, tăng số lượng người say rượu bị bắt tại chốt
                if idx == pol:
                    nCatches = nCatches + 1
                #print(path)
        #print("catch: {}/{}".format(nCatches, (len(restaurants)*nSamplings)))

        # cập nhật tỉ lệ bắt tại chốt pol
        catchRates.append(1.0*nCatches/(len(restaurants)*nSamplings))
    print(catchRates)
