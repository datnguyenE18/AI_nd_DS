/*Not solution*/
#include<bits/stdc++.h>
using namespace std;

/* Giả sử giao điểm giữa 2 đường chéo của hình vuông có tọa độ(0, 0)*/
double d = 20, // Cạnh hình vuông
	R = 3, // Bán kính hình tròn
	pi = 0,
	x = 5, y = -3; // Hoành và tung độ tâm hình tròn

int tries = 1000000, // Số lần thử
	inside = 0;

bool check(double a, double b) {
	if ( sqrt(pow(( a - x ), 2) + pow(( b - y ), 2)) <= R )
		return true;
	return false;
}

double random(double min, double max) {
	double tmp = rand() / (double) RAND_MAX;
	return min + tmp * ( max - min );
}

int main() {
	int rand_x = 0, rand_y = 0;
	double lmt = d / 2;
	srand(time(NULL));

	for ( int i = 0; i < tries; i++ ) {
		rand_x = random(-lmt, lmt);
		rand_y = random(-lmt, lmt);

		if ( check(rand_x, rand_y) )
			inside++;
	}

	pi = ( d * d * ( (double) inside / tries ) ) / ( R * R );
	cout << pi;
}

// KQ: 3.33596