#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <set>

using namespace std;

struct Comparator{
    bool operator() (const pair<int, set<int> >& lhs, const pair<int, set<int> >& rhs){
        return lhs.second.size() > rhs.second.size()
                || (lhs.second.size() == rhs.second.size() && lhs.first > rhs.first);
    }
};

class priority_graph : public priority_queue<pair<int, set<int> >, vector<pair<int, set<int> > >, Comparator >
{
private:
    set<int> nodes;       // for optimization reasons
public:
    void addEdge(int node_i, int node_j){
        bool found1 = false;
        bool found2 = false;
        for(auto it = this->c.begin(); it < this->c.end(); ++it)
        {
            if(it->first == node_i){
                it->second.insert(node_j);
                found1 = true;
            }
            if(it->first == node_j){
                it->second.insert(node_i);
                found2 = true;
            }
            if(found1 && found2) break;
        }
        make_heap(this->c.begin(), this->c.end(), this->comp);
    }
    set<int> getNeighbors(int node){
        set<int> res;
        for(auto it = this->c.begin(); it < this->c.end(); ++it){
            if(it->first == node){
                return it->second;
            }
        }
        return res;
    }
    void addNode(int id){
        auto res = nodes.insert(id);
        if(res.second == true)
            this->push(pair<int, set<int> >(id, set<int>()));
    }
    bool removeNode(int node, vector<int>& removed){
        nodes.erase(node);
        bool res;
        auto it = this->c.begin();
        for(; it < this->c.end(); ++it)
        {
            if(it->first == node){
                break;
            }
        }
        if (it != this->c.end()) {
            this->c.erase(it);
            make_heap(this->c.begin(), this->c.end(), this->comp);
            removed.push_back(node);
            res = true;
        }
        else{
            cout << "failed to remove node " << node << endl;
            return false;
        }

        it = this->c.begin();
        for(; it < this->c.end(); ++it)
        {
            it->second.erase(node);
        }

        return res;
    }
    set<int> getNodes(){
       return nodes;
    }
};
