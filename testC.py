def calculate_polygon_area(vertices):
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i+1) % n]  # Lấy đỉnh kế tiếp (hoặc đỉnh đầu tiên nếu i là đỉnh cuối cùng)
        area += x1*y2 - x2*y1
    return abs(area) / 2.0
vertices = [[59, -40],[-10,-40],[-20, -79], [-63, 24], [-31, 61], [52, 26]]
area = calculate_polygon_area(vertices)
print("Diện tích đa giác:", area)
