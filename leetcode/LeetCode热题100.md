## 哈希

[1. 两数之和 - 力扣（LeetCode）](https://leetcode.cn/problems/two-sum/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer,Integer> val2Index = new HashMap<>();
        // 遍历每个元素
        for(int i=0;i<nums.length;i++){
            int x = nums[i];
            // 如果之前遍历的元素存在值为target-x的，直接返回结果
            if(val2Index.containsKey(target - x)){
                return new int[]{i,val2Index.get(target - x)};
            }
            // 记录当前元素的值和下标
            val2Index.put(x,i);
        }
        return null;
    }
}
```

[49. 字母异位词分组 - 力扣（LeetCode）](https://leetcode.cn/problems/group-anagrams/?envType=study-plan-v2&envId=top-100-liked)

思路：题目要将字符串数组中的字母异位词（由重新排列源单词所有字母得到的一个新单词）组合在一起，并返回结果。如何判断两个单词是异位词呢，可以对单词进行编码，一共有26个字母，因此可以使用code[]数组统计每个字母的出现次数，将code数组转为字符串即为编码值。最后使用HashMap存储结果，键为对应异位词的编码值，值为编码值为对应编码值的所有字符串列表。

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        // 编码-分组映射
        Map<String,List<String>> code2Group=new HashMap<>();
        // 遍历每一种编码，将相同编码的元素添加到同一组中
        for(String str:strs){
            String code=encode(str);
            code2Group.putIfAbsent(code,new ArrayList<>());
            code2Group.get(code).add(str);
        }
        List<List<String>> res=new ArrayList<>();
        for(List<String> group:code2Group.values()){
            res.add(group);
        }
        return res;
    }
    //利用每个字符的出现次数进行编码
    String encode(String str){
        char[] code=new char[26];
        for(char c:str.toCharArray()){
            code[c-'a']++;
        }
        return new String(code);
    }
}
```

[128. 最长连续序列 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：

- 将所有元素添加到集合set中，便于判断该元素是否存在。
- 遍历所有的元素，对于每个元素，循环查找以当前元素开始的连续元素，并记录连续序列的长度
- 设当前元素值为x，如果存在值为x-1的元素，直接跳过（不是连续序列的开始位置）

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        //将所有元素添加到集合，便于查看是否存在
        Set<Integer> set=new HashSet<>();
        for(int x:nums){
            set.add(x);
        }
        int res=0;
        //寻找每一个序列
        for(int x:set){
            // 存在比当前值x更小的，不是序列开始位置
            if(set.contains(x-1)) continue;
            int len=1;
            // 循环查找比当前元素大1的元素
            while(set.contains(x+1)){
                len++;
                x++;
            }
            res=Math.max(res,len);
        }
        return res;
    }
    
}
```

思路二：

将数组去重并排序，遍历数组，每次以当前位置i作为连续序列的起始位置，并记录连续序列的长度，再将指针移到该连续序列的末尾，开始查找下一个连续序列。

```java
class Solution {
	public int longestConsecutive(int[] nums) {
        //将数组去重
        Set<Integer> s = new HashSet<>();
        for(int x:nums){
            s.add(x);
        }
        Integer[] nums2 = s.toArray(new Integer[0]);
        //将数组排序
        Arrays.sort(nums2);
        int n = nums2.length;
        int i=0;
        int len = 0;
		//遍历每个元素
        while(i<n){
            int j=i;
            //查找以nums[i]为起点的连续序列
            while(j+1<n && nums2[j+1]-nums2[j]==1) j++;
            len = Math.max(len,j-i+1);
            //移到下一个连续序列的起始位置
            i = j+1;
        }
        return len;
    }
}
```

## 双指针

[283. 移动零 - 力扣（LeetCode）](https://leetcode.cn/problems/move-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public void moveZeroes(int[] nums) {
        int i = 0;
        // 将非0数字移到前面
        for(int j=0;j<nums.length;j++){
            if(nums[j] != 0){
                nums[i++] = nums[j];
            }
        }
        // 将后面的元素全部置0
        while(i<nums.length) nums[i++] = 0;
    }
}
```

[11. 盛最多水的容器 - 力扣（LeetCode）](https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int maxArea(int[] height) {
        int left=0,right=height.length-1;
        int res=0;
        while(left<right){
            int t=Math.min(height[left],height[right])*(right-left);
            res=Math.max(res,t);
            if(height[left]<height[right]) left++;
            else right--;
        }
        return res;
    }
}
```

[15. 三数之和 - 力扣（LeetCode）](https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：首先将数组排序，枚举三元组的前2个元素的位置i, j，并在区间 [j+1,n-1] 二分查找第三个元素。由于结果不能重复，需要使用集合保存结果。

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Set<List<Integer>> res = new HashSet<>();
        //将数组排序
        Arrays.sort(nums);
        int n = nums.length;
        //枚举三元组前2个元素
        for(int i=0;i<n-2;i++){
            for(int j=i+1;j<n-1;j++){
                //二分查找第3个元素
                int k = bs(nums,j+1,n-1,-(nums[i]+nums[j]));
                if(k!=-1) res.add(new ArrayList<>(Arrays.asList(new Integer[]{nums[i],nums[j],nums[k]})));
            }
        }
        return new ArrayList<List<Integer>>(res);
    }

    int bs(int[] nums,int l,int r,int target){
        while(l<=r){
            int mid = (l+r)>>1;
            if(nums[mid] == target) return mid;
            else if(nums[mid] > target) r = mid-1;
            else l = mid+1;
        }
        return -1;
    }
}
```

思路二：

直接转换为nSum问题，遍历n元组开始元素的位置i, 递归求目标值为 target-nums[i] 的 (n-1) 元组，当n=2时，直接使用双指针求出结果即可。

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        // 将数组排序
        Arrays.sort(nums);
        return nSum(nums,3,0,0);
    }

    private List<List<Integer>> nSum(int[] nums,int n,int start,int target){
        int sz = nums.length;
        List<List<Integer>> res = new ArrayList<>();
        // 至少是 2Sum，且搜索区间大小sz-start不小于n
        if(n<2 || sz-start<n) return res;
        // base case
        if(n==2){
            int lo = start,hi = sz-1;
            while(lo < hi){
                int left = nums[lo],right = nums[hi];
                int sum = left + right;
                if(sum < target){ //和较小，右移左指针
                    while(lo < hi && nums[lo] == left) lo++;
                }else if(sum > target){ //和较大，左移右指针
                    while(lo < hi && nums[hi] == right) hi--;
                }else{
                    //记录二元组
                    res.add(new ArrayList(Arrays.asList(new Integer[]{left,right})));
                    //更新左右指针的位置
                    while(lo < hi && nums[lo] == left) lo++;
                    while(lo < hi && nums[hi] == right) hi--;
                }
            }
        }else{
            // 遍历元组的开始位置
            for(int i=start;i<sz;i++){
                // 递归计算子结果
                List<List<Integer>> sub = nSum(nums,n-1,i+1,target-nums[i]);
                // 将开始元素添加到子结果中
                for(List<Integer> list:sub){
                    list.add(nums[i]);
                    res.add(list);
                }
                // 跳过重复元素
                while(i < sz-1 && nums[i] == nums[i+1]) i++;
            }
        }
        
        return res;
    }
}
```

[42. 接雨水 - 力扣（LeetCode）](https://leetcode.cn/problems/trapping-rain-water/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：

每个柱子可以接到的雨水为当前柱子左边柱子的最大高度和右边柱子的最大高度的较小值减去当前柱子的高度，即：$water[i] = min(max(height[0..i],height[i..n-1]))-height[i]$，因此，直接计算出每个柱子左右两边的最大柱子高度lmax[i], rmax[i]，最后计算每个柱子可以接的雨水即可。

```java
class Solution {
    public int trap(int[] height) {
        int n=height.length;
        // lmax[i]: [0..i]区间的柱子高度的最大值
        int[] lmax = new int[n];
        // rmax[i]: [i..n-1]区间的柱子高度的最大值
        int[] rmax = new int[n];
        lmax[0] = height[0]; 
        rmax[n-1] = height[n-1];
        for(int i=1;i<n;i++){
            lmax[i] = Math.max(lmax[i-1],height[i]);
        }
        for(int i=n-2;i>=0;i--){
            rmax[i] = Math.max(rmax[i+1],height[i]);
        }
        int res = 0;
        // 计算每个柱子上可以接的雨水
        for(int i=0;i<n;i++){
            res += Math.min(lmax[i], rmax[i]) - height[i];
        }
        return res;
    }
}
```

思路二：

根据每个柱子接雨水的高度公式可以看出，当前柱子能接的水量是由左右最高柱子高度的较小值决定的，因此可以使用双指针left, right，在计算lmax[0..left]和rmax[right..n-1]的同时，计算每个柱子可以接的雨水：

- 若left < right：此时rmax[right..n-1] < rmax[left..n-1]，能接的雨水由lmax决定，res+=lmax-heigth[left]
- 若left > right：此时lmax[0..left] < lmax[0..right]，能接的雨水由rmax决定，res+=rmax-heigth[right]

```java
class Solution {
    public int trap(int[] height) {
        int n=height.length;
        int lmax=0,rmax=0;
        int left=0,right=n-1;
        int res=0;
        // 注意此处可以为<，因为最终height[left]=height[right]=maxHeight
        while(left<=right){
            lmax=Math.max(lmax,height[left]);
            rmax=Math.max(rmax,height[right]);
            if(lmax<rmax){
                res+=lmax-height[left];
                left++;
            }else{
                res+=rmax-height[right];
                right--;
            }
        }
        return res;
    }
}
```

## 滑动窗口

[3. 无重复字符的最长子串 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-substring-without-repeating-characters/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        // 定义窗口
        Map<Character, Integer> window = new HashMap<>();
        
        int left = 0, right = 0;
        int res = 0;
        while(right < s.length()){
            char c = s.charAt(right++);
            // 进行窗口内数据的一系列更新
            window.put(c, window.getOrDefault(c, 0) + 1);
            // 判断左侧窗口是否要收缩
            while(window.get(c) > 1){
                char d = s.charAt(left++);
                // 进行窗口内数据的一系列更新
                window.put(d, window.get(d) - 1);
            }
            // 更新结果
            res = Math.max(res, right-left);
        }
        return res;
    }
}
```

[438. 找到字符串中所有字母异位词 - 力扣（LeetCode）](https://leetcode.cn/problems/find-all-anagrams-in-a-string/?envType=study-plan-v2&envId=top-100-liked)

思路：直接使用滑动窗口模板即可，使用need保存模式串p中所有字符，每次将一个字符添加到窗口中，当窗口中字符数量大于模式串p的长度时需要收缩窗口，使用count记录此时满足数量要求的字符数。收缩后窗口的长度即与模式串p的长度相等，如果满足要求的字符数等于need中的字符数，则找到了一个合法结果。

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        Map<Character, Integer> window = new HashMap<>();
        Map<Character, Integer> need = new HashMap<>();
        for(char c:p.toCharArray()) need.put(c, need.getOrDefault(c, 0) + 1);

        // 记录窗口中满足数量要求的字符数
        int count = 0;
        int left = 0, right = 0;
        // 记录结果
        List<Integer> res = new ArrayList<>();
        while(right < s.length()){
            char c = s.charAt(right++);
            // 进行窗口内的一些列更新
            if(need.containsKey(c)){
                window.put(c, window.getOrDefault(c, 0) + 1);
                if(window.get(c).equals(need.get(c))) count++;
            }
            // 判断左侧窗口是否要收缩
            while(right - left > p.length()){
                char d = s.charAt(left++);
                // 进行窗口内的一些列更新
                if(need.containsKey(d)){
                    if(window.get(d).equals(need.get(d))) count--;
                    window.put(d, window.get(d) - 1);
                }
            }
            // 满足数量要求的字符与模式串中相等，记录结果
            if(count == need.size()) res.add(left);
        }
        return res;
    }
}
```

## 子串

[560. 和为 K 的子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/subarray-sum-equals-k/?envType=study-plan-v2&envId=top-100-liked)

思路一：

题目即求元素之和为k的子数组，设子数组所在的区间为区间为$[i, j]$，则可以枚举每一个区间，统计满足条件的即可。可以现枚举区间的结束位置$j$, 再枚举开始位置$i$，开始位置可以反向遍历，边遍历边计算元素和。

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        int n = nums.length;
        int res = 0;
        //枚举结束位置
        for(int i=0;i<n;i++){
            int sum = 0;
            //枚举开始位置
            for(int j=i;j>=0;j--){
                sum += nums[j];
                if(sum == k) res++;
            }
        }
        return res;
    }
}
```

思路二：

- 可以先计算数组的前缀和，区间$[i,j]$的子数组元素之和即为$k=pre[j]-pre[i-1]$，则对于结束位置$j$，需要找到满足$pre[i-1]=pre[j]-k$的所有$i$。
- 可以使用哈希表map保存当前枚举结束位置$j$左侧各前缀和的出现次数，则当结束位置为$j$时满足条件的子数组个数为$map[pre[j]-k]$，累加到结果即可。
- 由于需要知道当前枚举结束位置前面特定前缀和的出现次数，需要边遍历边计算前缀和，即使用变量pre保存当前前缀和$nums[0..j]$，同时将前缀和保存在map映射中。

```java
class Solution {
    // pre[i]-pre[j-1] = k
    public int subarraySum(int[] nums, int k) {
        int n = nums.length;
        // map[x]=n表示前缀和数组中当前值为x的元素数量为n
        Map<Integer, Integer> map = new HashMap<>();
        int res = 0;
        // 记录当前的前缀和
        int pre = 0;
        // 初始化（前缀和为0的元素个数为1）
        map.put(0, 1);
        for(int i=0;i<n;i++){
            pre += nums[i];
            if(map.containsKey(pre - k)){
                res += map.get(pre - k);
            }
            map.put(pre, map.getOrDefault(pre, 0) + 1);
        }
        return res;
    }
}
```

[239. 滑动窗口最大值 - 力扣（LeetCode）](https://leetcode.cn/problems/sliding-window-maximum/?envType=study-plan-v2&envId=top-100-liked)

思路：

题目要求固定长度的窗口中的最大值，可以使用单调队列实现。这里使用LinkedList实现单调递减队列：

- 在添加元素时，将队尾比自身小的元素都弹出
- 在弹出一个元素时，若队首元素等于该元素则弹出，否则说明该元素在添加元素时已经被弹出，不做处理
- 在求窗口中的最大值时直接取队首元素即可

```java
class Solution {
    // 单调队列
    static class MonotonicQueue{
        LinkedList<Integer> maxq;
        public MonotonicQueue(){
            this.maxq=new LinkedList<>();
        }

        void push(int x){
            // 将前面小于x的元素弹出
            while(!maxq.isEmpty()&&maxq.getLast()<x) maxq.pollLast();
            maxq.addLast(x);
        }

        void pop(int x){
            // 当队头元素为x时，删除
            if(maxq.getFirst()==x) maxq.pollFirst();
        }

        int max(){
            // 队头元素为最大值
            return maxq.getFirst();
        }
    }
    public int[] maxSlidingWindow(int[] nums, int k) {
        MonotonicQueue window=new MonotonicQueue();
        int n=nums.length;
        int[] res=new int[n-k+1];
        for(int i=0;i<n;i++){
            if(i<k-1){
                // 先将前k-1个元素填入窗口
                window.push(nums[i]);
            }else{
                // 窗口开始向前滑动
                window.push(nums[i]);
                // 获取窗口最大元素
                res[i-k+1]=window.max();
                // 将离开窗口的元素删除
                window.pop(nums[i-k+1]);
            }
        }
        return res;
    }
}
```

[76. 最小覆盖子串 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-window-substring/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public String minWindow(String s, String t) {
        Map<Character, Integer> window = new HashMap<>();
        Map<Character, Integer> need = new HashMap<>();
        for(char c:t.toCharArray()) need.put(c, need.getOrDefault(c, 0)+1);

        int valid = 0;
        int left = 0,right = 0;
        // 记录最小覆盖字串的起始位置及长度
        int start = 0, len = Integer.MAX_VALUE;
        while(right < s.length()){
            char c = s.charAt(right++);
            // 进行窗口内数据的一系列更新
            if(need.containsKey(c)) {
                window.put(c, window.getOrDefault(c, 0)+1);
                if(window.get(c).equals(need.get(c))) valid++;
            }

            // 判断左侧窗口是否要收缩
            while(valid == need.size()){
                if(right-left < len){
                    start = left;
                    len = right - left;
                }
                char d = s.charAt(left++);
                // 进行窗口内数据的一系列更新
                if(need.containsKey(d)) {
                    if(window.get(d).equals(need.get(d))) valid--;
                    window.put(d, window.get(d)-1);
                }
            }
        }
        // 返回最小覆盖子串
        return len == Integer.MAX_VALUE?"":s.substring(start, start+len);
    }
}
```

