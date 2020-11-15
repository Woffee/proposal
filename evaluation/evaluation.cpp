#include <iostream>

#include "armadillo"

using namespace arma;
using namespace std;

mat data_o;
mat data_e;
mat x_near;

float distance(const mat& row1, const mat& row2)
{
    float s = 0;
    for(int i=0; i<5; i++){
        float a = row1(0, i);
        float b = row2(0, i);
        s = s + (a-b) * (a-b);
    }
    return s;
}

// 在 data_o 中找出 与 row 最接近的，返回索引
int find_near(const mat& row)
{
    float dist = distance( row, data_o.row(0) );
    int near = 0;

    int j=1;
    for( ; j< data_o.n_rows; j++  ){
        float d = distance(  row,  data_o.row(j)  );
        if( d<dist ){
            dist = d;
            near = j;
        }
        if( d<0.001 ){
            return near;
        }
    }
    return near;
}

float dot_col( int c)
{
    float tmp=0;
    for(int i=0; i<x_near.n_rows; i++){
        float aa = x_near(i, c);
        float bb = data_e(i, c);
        tmp += aa * bb;
    }
    return tmp;
}

float diff_col(int c)
{
    float res=0;
    for(int i=0; i<x_near.n_rows; i++){
        float aa = x_near(i, c);
        float bb = data_e(i, c);
        float t = aa - bb;
        res += t * t;
    }
    return res;
}

float cal_accuracy(int c)
{
    float near_dot = dot_col(c);
    float t = diff_col(c);
    return near_dot / (t + near_dot);
}

int main(int argc, char** argv)
{
    mat a;
    mat b;
    a << 1.0 << 2.0 << 3.0 <<endr;
    b << 4.0 << 5.0 << 6.0 <<endr;
    mat c = a - b;
    c.print();
    return 0;



    cout<<"loading data_e: "<<argv[1]<<endl;
    cout<<"loading data_o: "<<argv[2]<<endl;



    data_e.load(argv[1]);
    data_o.load(argv[2]);
    cout<<"data_e: "<<data_e.n_rows<<" x "<<data_e.n_cols<<endl;
    cout<<"data_o: "<<data_o.n_rows<<" x "<<data_o.n_cols<<endl;


    for(int i=0; i<data_o.n_rows; i++){
        if(i%100==0) cout<<"now: "<<i<<" / " << data_o.n_rows <<endl;

        if( distance(data_o.row(i), data_e.row(i)) == 0 ){
            x_near.insert_rows( x_near.n_rows, data_e.row(i) );
        }else{
            int j = find_near( data_e.row(i) );
            x_near.insert_rows( x_near.n_rows, data_e.row(j) );
        }
    }

    cout<<"x_near: "<<x_near.n_rows<<" x "<<x_near.n_cols<<endl;

    float sum_e = 0;
    for(int i=0; i < data_e.n_cols; i++){
        float e = cal_accuracy(i);
        sum_e += e;
        cout<<"e"<<i<<" "<<e<<endl;
    }

    float acc = sum_e / data_e.n_cols;
    cout<<argv[2]<<" accuracy2: "<< acc <<endl;
}