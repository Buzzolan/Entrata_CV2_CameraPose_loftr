function v = vec(A); 
%VEC riorganizza gli elementi di una matrice in un vettore 
%    colonna.
%
%v = vec(A) data una matrice A (n x m) ritorna un vettore colonna
%di n x m elementi, i cui elementi sono l'unione delle colonne 
%della matrice A.

v = reshape(A,size(A,1)*size(A,2),1);
