close all;  
clear all;  

%%%% On a vérifié au préalable avec issymetric(A) 
%%% que les 5 matrices ci-dessous sont symétriques
load mat0.mat;
%load mat1.mat;
%load mat2.mat;
%load mat3.mat;
%load bcsstk27.mat;  

n = size(A, 1); 

b = [1:n]';  

% resolution du systeme Ax = b
x_sol = A\b;  

% Factorisation de Cholesky
[L, flag] = chol(A, "lower");

% Vérifie si la matrice A est semi-définie positive
if flag == 0
    fprintf("A semi définie positive\n");
else
    fprintf("Attention, A n'est pas semi définie positive !\n");
end

% trace le graphique de la diag supérieure 
% de A et de L
spy(triu(A)+L);
fprintf("Nombre de non zéros de A : %4d \n", 2*nnz(A)); 
fprintf("Nombre d'opérations pour la phase de résolution : %4d \n", 4*nnz(L)-2*n);  
% resolution du systeme L*y = b
y = L\b;
% resolution du systeme L'*x1 = y
x1 = L'\y;

% Erreur normwise sur le résidu
norm(b-A*x1)/norm(b);

% Erreur normwise sur la solution
norm(x_sol - x1) / norm(x_sol);

% Visualisation de la matrice originale
subplot(2,3,1);
spy(A);
title('Matrice originale A');

% factorisation symbolique de A.
[count,h,parent,post,R] = symbfact(A);
ALU=R+R';
subplot(2,3,2);
spy(ALU);
title('Facteurs de A');
fillin=nnz(ALU)-nnz(A);

% Visualisation du fill-in
C=spones(A);
CLU=spones(ALU);
FILL=CLU-C;
subplot(2,3,3);
spy(FILL);
title('Remplissage sur A');

% Permutation de la matrice A 
% selon plusieurs schémas

P = symamd(A);
%P = symrcm(A);
%P = amd(A);
%P = colamd(A);
%P = colperm(A);
B = A(P,P);

subplot(2,3,4);
spy(B);
title('Matrice permutée B');

% factorisation symbolique de B.
[count,h,parent,post,R] = symbfact(B);
BLU=R+R';
subplot(2,3,5)
spy(BLU);
title('Facteurs de B');
fillin=nnz(BLU)-nnz(B);

% Visualisation du fill-in 
% sur la matrice permutée B.
C=spones(B);
CLU=spones(BLU);
FILL=CLU-C;
subplot(2,3,6);
spy(FILL);
title('Remplissage sur B');

% Factorisation de Cholesky de B
L = chol(B, 'lower');

y = L\b(P);
x2 = L'\y;
% permutation de  x2 pour 
% retrouver l'ordre d'origine
x2(P) = x2;

%%% Erreur normwise sur le résidu
normwise_res = norm(b-A*x2)/norm(b);

%%% Erreur normwise sur la solution
normwise_sol = norm(x_sol - x2) / norm(x_sol);


fprintf("Erreur normwise sur le résidu de la matrice permutée : %4d \n", normwise_res);  
fprintf("Erreur normwise sur la solution : %4d \n", normwise_sol);  
fprintf("Nombre d'opérations pour la phase de résolution : %4d \n", 4*nnz(L)-2*n); 
    