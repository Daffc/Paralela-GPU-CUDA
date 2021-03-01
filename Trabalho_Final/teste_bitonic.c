#include <stdlib.h>
#include <stdio.h>

int main (){

    int arr []= {5, 5, 6, 2, 1, 3, 4, 1};
    int tmp;
    int n = 8;
    int aux;

    int i,j,k;
    for (k=2; k <= n; k=2*k) {
      for (j = k>>1; j > 0; j = j>>1) {
        for (i=0; i < n ; i++) {
          int ixj = i^j;
          if ((ixj) > i) {
            if ((i&k) == 0 && arr[i]>arr[ixj]){
                aux = arr[i];
                arr[i] = arr[ixj];
                arr[ixj] = aux;
            } 
            if ((i&k)!=0 && arr[i]<arr[ixj]){
                aux = arr[i];
                arr[i] = arr[ixj];
                arr[ixj] = aux;
            }
          }
        }
      }
    }

    for (i = 0; i < n; i++)
        printf("%d", arr[i]);

    printf("\n");

    return 0;
}