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

## 普通数组

[53. 最大子数组和 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-subarray/?envType=study-plan-v2&envId=top-100-liked)

思路一：

使用分治算法，将数组分为两部分，最大的子数组和可能出现在左数组、右数组或左右两数组各取一部分，前2种情况可以直接递归求，第三种枚举子数组的开始位置和结束位置求，最终取三者的最大值即可，时间复杂度为O(n^3)。

```java
class Solution {
    int[] pre;
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        pre = new int[n+1];
        for(int i=0;i<n;i++) pre[i+1] = pre[i] + nums[i];
        return maxSubArray(nums, 0, n-1);
    }
    int maxSubArray(int[] nums, int lo, int hi){
        if(lo == hi) return nums[lo];
        int mid = (lo + hi) >> 1;
        int maxLeft = maxSubArray(nums, lo, mid);
        int maxRight = maxSubArray(nums, mid+1, hi);
        int res = Math.max(maxLeft, maxRight);
        for(int i=0;i<=mid;i++){
            for(int j=mid+1;j<=hi;j++){
                res = Math.max(res, pre[j+1] - pre[i]);
            }
        }
        return res;
    }
}
```

思路二：

使用动态规划，定义dp[i]为以nums[i]结尾的子数组的最大和，则对于nums[i]有两种选择：

- 拼接在前面的以nums[i-1]结尾的子数组，则dp[i] = dp[i-1] + nums[i]
- 独自成为一个子数组，则dp[i] = nums[i]

最终枚举子数组的结束位置，取最大值即可。

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        // dp[i]表示以nums[i]结尾的子数组的最大和
        int[] dp = new int[n];
        dp[0] = nums[0];
        for(int i=1;i<n;i++){
            dp[i] = Math.max(nums[i], dp[i-1] + nums[i]);
        }
        int res = Integer.MIN_VALUE;
        // 枚举每一个结尾
        for(int i=0;i<n;i++){
            res = Math.max(res, dp[i]);
        }
        return res;
    }
}
```

[56. 合并区间 - 力扣（LeetCode）](https://leetcode.cn/problems/merge-intervals/?envType=study-plan-v2&envId=top-100-liked)

思路：先将区间按照开始位置升序排序，对于每个区间[left, right]，设已合并的最后一个区间为[start, end]。若left <= end，则可以与该区间合并，更新最后一个合并区间的结束位置为Math.max(end, right)，否则需要作为一个新的合并区间。

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        // 将区间按开始位置升序排序
        Arrays.sort(intervals, (o1, o2) -> o1[0]-o2[0]);
        List<int[]> res = new ArrayList<>();
        // 加入第一个区间
        res.add(intervals[0]);
        for(int i = 1;i<intervals.length;i++){
            int[] cur = intervals[i];
            int[] last = res.get(res.size()-1);
            if(cur[0] <= last[1]){
                // 当前区间可以和最后一个区间合并，更新最后一个区间的结束位置
                last[1] = Math.max(last[1], cur[1]);
            }else{
                // 不能合并，将当前区间添加到res中
                res.add(cur);
            }
        }
        return res.toArray(new int[0][0]);
    }
}
```

[189. 轮转数组 - 力扣（LeetCode）](https://leetcode.cn/problems/rotate-array/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：使用额外的数组

使用一个额外的数组，将每个元素移到空数组对应位置，再将空数组复制回原数组即可。

```java
class Solution {
    public void rotate(int[] nums, int k) {
        int n=nums.length;
        k%=n;
        int[] temp=new int[n];
        for(int i=0;i<n;i++) temp[(i+k)%n]=nums[i];
        for(int i=0;i<n;i++) nums[i]=temp[i];
    }
}
```

思路二：数组翻转

先将整个数组翻转，再分别将区间 [0, k-1] 和 [k, n-1] 的元素翻转，得到的结果即为轮转 k 步后的数组。

```java
class Solution {
    public void rotate(int[] nums, int k) {
        int n=nums.length;
        k %= n;
        reverse(nums, 0, n-1);
        reverse(nums, 0, k-1);
        reverse(nums, k, n-1);
    }
    void reverse(int[] nums, int l, int r){
        while(l<r){
            int temp = nums[l];
            nums[l] = nums[r];
            nums[r] = temp;
        }
    }
}
```

思路三：环状替换

对于开始位置 start 的元素，需要被移动到 (start+k)%n 位置，使用 temp 保存被置换的元素，下一次将 temp 的值设置给下一个位置，依此类推，直到回到开始位置。但是此时还是有很多元素是没有遍历到的，因此需要从开始位置的下一个位置 start+1 开始继续替换，直到所有元素都被替换。

设每轮经过的圈数为x，每一轮遍历到的元素数量为 y，则有 xn = yk，显然 yk 是n和k的公倍数，又因为y尽可能小，故 yk 是n和k的最小公倍数，即  yk = nk/gcd(n, k)，n/y=gcd(n, k)，则遍历的轮数为gcd(n, k)。

```java
class Solution {
    public void rotate(int[] nums, int k) {
        int n=nums.length;
        k %= n;
        int count = gcd(n, k);
        for(int start=0;start<count;start++){
            int i = start;
            int prev = nums[start];
            do{
                int j = (i+k)%n;
                int temp = nums[j];
                nums[j] = prev;
                prev = temp;
                i = j;
            }while(i != start);
        }
    }

    int gcd(int x, int y){
        return y == 0?x:gcd(y, x%y);
    }
}
```

[238. 除自身以外数组的乘积 - 力扣（LeetCode）](https://leetcode.cn/problems/product-of-array-except-self/?envType=study-plan-v2&envId=top-100-liked)

思路一：

定义数组L, R分别表示 nums[i] 左侧和右侧的元素乘积，最终计算遍历数组计算每个元素除自身之外的元素乘积，时间复杂度为O(n^2)，空间复杂度为O(n)。

```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        // L[i]:nums[i]左侧元素的乘积
        // R[i]:nums[i]右侧元素的乘积
        int[] L = new int[n], R = new int[n];
        L[0] = 1;
        for(int i=1;i<n;i++){
            L[i] = L[i-1] * nums[i-1];
        }
        R[n-1] = 1;
        for(int i=n-2;i>=0;i--){
            R[i] = R[i+1] * nums[i+1];
        }

        int[] res = new int[n];
        for(int i = 0;i<n;i++){
            res[i] = L[i] * R[i];
        }
        return res;
    }
}
```

思路二：

直接先使用 res 数组计算L数组，然后计算 R 数组，只是使用 R 变量保存，边遍历边计算结果，时间复杂度为O(n^2)，空间复杂度为O(1)。

```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        res[0] = 1;
        for(int i=1;i<n;i++){
            res[i] = res[i-1] * nums[i-1];
        }
        int R = 1;
        for(int i = n-2;i>=0;i--){
            R *= nums[i+1];
            res[i] *= R;
        }
        return res;
    }
}
```

[41. 缺失的第一个正数 - 力扣（LeetCode）](https://leetcode.cn/problems/first-missing-positive/?envType=study-plan-v2&envId=top-100-liked)

思路：

显然未出现的第一个正数的范围为 [1, n+1]，因此可以将范围 [1, n] 的元素 x 进行标记，最终找出区间 [1, n] 中未被标记的最小元素即为结果，如果都被标记则结果为 n+1。可以借助原始数组，先将不在范围 [1, n]的元素都置为 n+1，对于范围 [1, n] 的元素x，将 arr[x-1] 置为 -arr[x-1]，最终找出数组中非负数的最小位置即为结果。

```java
class Solution {
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        // 将[,0]范围的元素都设为n+1
        for(int i=0;i<n;i++){
            if(nums[i]<=0) nums[i] = n+1;
        }
        // 将[1,n]范围的元素x对应的位置x-1打上标记
        for(int i=0;i<n;i++){
            int num = Math.abs(nums[i]);
            if(num <= n){
                nums[num-1] = -Math.abs(nums[nums-1]);
            }
        }
        // 没有的打上标记的第一个元素即为结果
        for(int i=0;i<n;i++){
            if(nums[i] > 0) return i+1;
        }
        return n+1;
    }
}
```

## 矩阵

[73. 矩阵置零 - 力扣（LeetCode）](https://leetcode.cn/problems/set-matrix-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：

先将值为0的元素坐标都保存到列表中，遍历列表中的坐标，将每个位置的横纵行元素都置为0，时间复杂度O(nm^2)。

```java
class Solution {
    public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        // 值为0的元素坐标
        List<int[]> list = new ArrayList<>();
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(matrix[i][j] == 0) list.add(new int[]{i,j});
            }
        }
        // 将每个坐标对应的行列置为0
        for(int[] tp:list){
            for(int i = 0;i<m;i++) matrix[i][tp[1]] = 0;
            for(int j = 0;j<n;j++) matrix[tp[0]][j] = 0;
        }
    }
}
```

思路2：

使用首行和首列保存对应列和对应行是否需要置0，并使用 flag_col0, flag_row0 记录首行和首列需要置0，先将中间元素对应位置置0，再将首行和首列置0。

```java
class Solution {
    public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        // 标记第一列和第一行是否需要置0
        boolean flag_col0 = false, flag_row0 = false;
        for(int i=0;i<m;i++){
            if(matrix[i][0] == 0){
                flag_col0 = true;
                break;
            } 
        }
        for(int j=0;j<n;j++){
            if(matrix[0][j] == 0){
                flag_row0 = true;
                break;
            }
        }

        // 标记需要置为0的行和列
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                if(matrix[i][j] == 0) {
                    matrix[0][j] = matrix[i][0] = 0;
                }
            }
        }

        // 遍历中间的元素并将对应元素置0
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                if(matrix[0][j] == 0 || matrix[i][0] == 0){
                    matrix[i][j] = 0;
                }
            }
        }

        // 置0首行和首列的
        if(flag_col0){
            for(int i=0;i<m;i++) matrix[i][0] = 0;
        }
        if(flag_row0){
            for(int j=0;j<n;j++) matrix[0][j] = 0;
        }
    }
}
```

[54. 螺旋矩阵 - 力扣（LeetCode）](https://leetcode.cn/problems/spiral-matrix/description/?envType=study-plan-v2&envId=top-100-liked)

思路：

记录当前未经过元素的上下左右边界，从矩阵左上角开始，依次向右下左上遍历，每遍历完一个方向，更新对应的边界值，当无法继续遍历时结束。

```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        // 定义边界
        int upper = 0, lower = m-1, left = 0, right = n-1;

        List<Integer> res = new ArrayList<>();
        while(true){
            for(int j=left;j<=right;j++) res.add(matrix[upper][j]); //向右移动直到最右
            if(++upper>lower) break;
            for(int i=upper;i<=lower;i++) res.add(matrix[i][right]); //向下
            if(--right<left) break;
            for(int j=right;j>=left;j--) res.add(matrix[lower][j]); //向左
            if(--lower<upper) break;
            for(int i=lower;i>=upper;i--) res.add(matrix[i][left]); //向上
            if(++left>right) break;
        }

        return res;
    }
}
```

[48. 旋转图像 - 力扣（LeetCode）](https://leetcode.cn/problems/rotate-image/?envType=study-plan-v2&envId=top-100-liked)

思路：

顺时针旋转矩阵90度后，矩阵的行都变成了列。将矩阵按照对角线翻转后，可以发现结果矩阵和旋转后的矩阵是沿竖线对称的，因此将翻转后的矩阵每一行转置即可得到旋转矩阵。

```java
class Solution {
    public void rotate(int[][] matrix) {
        int n=matrix.length;
        // 沿对角线翻转矩阵
        for(int i=0;i<n;i++){
            for(int j=i;j<n;j++){
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }

        // 翻转矩阵的每一行
        for(int i=0;i<n;i++){
            for(int j=0;j<n/2;j++){
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][n-1-j];
                matrix[i][n-1-j] = temp;
            }
        }
    }
}
```

[240. 搜索二维矩阵 II - 力扣（LeetCode）](https://leetcode.cn/problems/search-a-2d-matrix-ii/?envType=study-plan-v2&envId=top-100-liked)

思路一：二分查找

由于矩阵的每一行都是有序的，因此可以对每一行进行二分查找，找到了结果即可返回。

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        //二分搜索每一行
        for(int[] row:matrix){
            if(row[0]>target) return -1;
            int idx = bs(row, target);
            if(idx != -1) return true;
        }
        return false;
    }

    int bs(int[] arr, int target){
        int left = 0, right = arr.length-1;
        while(left <= right){
            int mid = (left + right) >> 1;
            if(arr[mid] == target) return mid;
            else if(arr[mid] > target) right = mid-1;
            else left = mid+1;
        }
        return -1;
    }
}
```

思路二：搜索

矩阵的每一行和每一列都是升序的，可以发现，在矩阵的右上角时，向左移动元素值会减小，向下移动时元素值会增大。因此可以从右上角开始搜索，若元素较大，向左移动，否则向下移动，直到找到目标值结束。

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        // 从右上角开始搜索
        int x = 0, y = n-1;
        while(x<m && y>=0){
            if(matrix[x][y] == target) return true;
            else if(matrix[x][y] > target) y--;
            else x++;
        }
        return false;
    }
}
```

## 链表

[160. 相交链表 - 力扣（LeetCode）](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=study-plan-v2&envId=top-100-liked)

思路:

为了找到链表的相交位置，可以分别将另一条链表拼接在本链表的后面，这样两条链表的长度就相等了。同时遍历两条链表，这样遇到相同的节点即为相交的位置。

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode pa = headA, pb = headB;

        while(pa != pb){
            // pa走一步，如果走到A链表末尾，转到B链表
            if(pa == null) pa = headB;
            else pa = pa.next;
            // pb走一步，如果走到B链表末尾，转到A链表
            if(pb == null) pb = headA;
            else pb = pb.next;
        }

        return null;
    }
}
```

[206. 反转链表 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-linked-list/?envType=study-plan-v2&envId=top-100-liked)

思路：

利用递归函数定义，递归调用反转函数，将以 head.next 节点为头结点的链表反转，返回子链表的头节点 last，再将头节点接到子链表末尾即可。

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null || head.next == null) return head;
        // 反转以头节点的下一个节点为首节点的子链表
        ListNode last = reverseList(head.next);
        // 将头节点接在子反转链表的结尾
        head.next.next = head;
        // 将头节点的next指针置空
        head.next = null; 
        return last;
    }
}
```

[234. 回文链表 - 力扣（LeetCode）](https://leetcode.cn/problems/palindrome-linked-list/?envType=study-plan-v2&envId=top-100-liked)

思路：

先使用快慢指针方法找到链表的中间位置，并将后半部分的链表反转，遍历前半部分和后半部分的链表，判断节点是否都相等即可。

```java
class Solution {
    public boolean isPalindrome(ListNode head) {
        ListNode fast=head,slow=head;
        // 使得slow指针到达中间位置
        while(fast!=null&&fast.next!=null){
            slow=slow.next;
            fast=fast.next.next;
        }
        // 如果fast不为null，说明链表长度为奇数，slow指针前进一步
        if(fast!=null) slow=slow.next;
        // 将链表后半部分反转
        ListNode q=reverse(slow);
        ListNode p=head;
        while(q!=null&&q.val==p.val){
            p=p.next;
            q=q.next;
        }
        if(q==null) return true;
        else return false;
    }

    // 将以head为头节点的链表反转
    ListNode reverse(ListNode head){
        ListNode pre=null,cur=head,nxt;
        while(cur!=null){
            nxt=cur.next;
            cur.next=pre;
            pre=cur;
            cur=nxt;
        }
        return pre;
    }
}
```

[141. 环形链表 - 力扣（LeetCode）](https://leetcode.cn/problems/linked-list-cycle/?envType=study-plan-v2&envId=top-100-liked)

思路：

使用快慢指针方法，若链表中存在环，则两指针一定会相交。

```java
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            // 快慢指针相遇，说明出现环
            if(slow == fast) return true;
        }
        return false;
    }
}
```

[142. 环形链表 II - 力扣（LeetCode）](https://leetcode.cn/problems/linked-list-cycle-ii/?envType=study-plan-v2&envId=top-100-liked)

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head, fast = head;
        // 使得快慢指针相遇
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(fast == slow) break;
        }
        // 不存在环，直接返回
        if(fast == null || fast.next == null) return null;
        // slow指针指向头节点
        slow = head;
        // slow指针和fast指针同时前进，直到相遇
        while(slow != fast){
            slow = slow.next;
            fast = fast.next;
        }

        // 相遇位置即为环的起点
        return slow;
    }
}
```

[142. 环形链表 II - 力扣（LeetCode）](https://leetcode.cn/problems/linked-list-cycle-ii/?envType=study-plan-v2&envId=top-100-liked)

思路：

设slow指针走了k步，fast指针走了2k步，相遇位置距离环起点x步，环的周长为n，则有k=cn。可以发现，从起点走k-x步到达环起点，而从相遇位置再走k-x步恰好也回到了环起点(x+k-x=cn)。因此先让slow, fast指针到达相遇位置，然后slow指针回到起点，再同时前进，相遇的位置即为环起点位置。

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head, fast = head;
        // 使得快慢指针相遇
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(fast == slow) break;
        }
        // 不存在环，直接返回
        if(fast == null || fast.next == null) return null;
        // slow指针指向头节点
        slow = head;
        // slow指针和fast指针同时前进，直到相遇
        while(slow != fast){
            slow = slow.next;
            fast = fast.next;
        }

        // 相遇位置即为环的起点
        return slow;
    }
}
```

[21. 合并两个有序链表 - 力扣（LeetCode）](https://leetcode.cn/problems/merge-two-sorted-lists/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        // 虚拟头节点
        ListNode dummy = new ListNode(), p = dummy;
        ListNode p1 = list1, p2 = list2;

        while(p1 != null && p2 != null){
            // 比较 p1 和 p2 两个指针
            if(p1.val <= p2.val) {
                p.next = p1;
                p1 = p1.next;
            }else{
                p.next = p2;
                p2 = p2.next;
            }
            // p 指针不断前进
            p = p.next;
        }
        if(p1 != null){
            p.next = p1;
        }
        if(p2 != null){
            p.next = p2;
        }

        return dummy.next;
    }
}
```

[2. 两数相加 - 力扣（LeetCode）](https://leetcode.cn/problems/add-two-numbers/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(), p = dummy;
        ListNode p1 = l1, p2 = l2;
        // 记录进位
        int carry = 0;
        // 开始执行加法，两条链表走完且没有进位才结束循环
        while(p1 != null || p2 != null || carry != null){
            // 先加上上次进位
            int val = carry;
            if(p1 != null){
                val += p1.val;
                p1 = p1.next;
            }
            if(p2 != null){
                val += p2.val;
                p2 = p2.next;
            }
            // 计算本次进位
            carry = val/10;
            val = val%10;
            // 构建新节点连接到结果链表结尾
            p.next = new ListNode(val);
            p = p.next;
        }
        // 返回结果链表头节点（去除虚拟头节点） 
        return dummy.next;
    }
}
```

[19. 删除链表的倒数第 N 个结点 - 力扣（LeetCode）](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/?envType=study-plan-v2&envId=top-100-liked)

思路：

要删除倒数第n个节点，先要找到倒数第 n+1 个节点。要找到倒数第 k 个节点，即顺数第 n-k+1 个节点，可以使用两个指针分别指向头节点，先让一个指针走 n-k 步，再让两个指针同时前进，当第一个指针到达末尾时终止，此时走了 n-k 步，第二个指针指向第 n-k+1 个节点，即倒数第 k 个节点。

```java
lass Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if(head.next == null) return null;
        ListNode dummy = new ListNode(-1, head);
        // 删除倒数第 n 个，要先找到倒数第 n+1 个节点
        ListNode preNode = findFromEnd(dummy, n+1);
        // 删除倒数第 n 个节点
        preNode.next = preNode.next.next;
        return dummy.next;
    }
    // 返回链表倒数第 k 个节点
    ListNode findFromEnd(ListNode head, int k){
        ListNode p1 = head, p2 = head;
        // p1 先走 k 步
        for(int i=0;i<k;i++) p1 = p1.next;
        // p1 和 p2 同时走 n-k 步
        while(p1 != null){
            p1 = p1.next;
            p2 = p2.next;
        }
        // p2 现在指向第 n-k+1 个节点（倒数第k个节点）
        return p2;
    }
}
```

[24. 两两交换链表中的节点 - 力扣（LeetCode）](https://leetcode.cn/problems/swap-nodes-in-pairs/?envType=study-plan-v2&envId=top-100-liked)

思路：

利用递归函数定义，先将前2个元素翻转，再递归调用翻转函数，将子链表两两交换，最后将前两个节点接到后面即可。

```java
class Solution {
    public ListNode swapPairs(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode first = head;
        ListNode second = head.next;
        ListNode others = head.next.next;
        // 先将前两个元素翻转
        second.next = first;
        // 递归调用，将剩下的链表节点两两翻转，接到后面
        first.next = swapPairs(others);
        // 返回新的头节点
        return second;
    }
}
```

[25. K 个一组翻转链表 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-nodes-in-k-group/?envType=study-plan-v2&envId=top-100-liked)

思路：

首先找到头节点后面的k个节点的位置b，先将前k个节点翻转，再递归调用翻转函数，将后面的链表翻转，并接到前面k个节点翻转后的链表后面。

```java
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode a = head, b = head;
        // 区间 [a, b) 包含 k 个待反转元素
        while(int i = 0;i<k;i++){
            // base case: 不足k个，不需要反转
            if(b == null) return head;
            b = b.next;
        }
        // 反转前 k 个元素
        ListNode newHead = reverse(a, b);
        // 递归反转后续链表并连接起来
        a.next = reverseKGroup(b, k);
        return newHead;
    }
    // 反转链表区间 [a,b) 的节点
    ListNode reverse(ListNode a, ListNode b){
        ListNode pre = null, cur = head, nxt;
        while(a != b){
            nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }
}
```

[138. 随机链表的复制 - 力扣（LeetCode）](https://leetcode.cn/problems/copy-list-with-random-pointer/?envType=study-plan-v2&envId=top-100-liked)

思路：

使用哈希表 originToClone 保存原节点和复制节点的映射，遍历一遍链表，创建复制节点，并保存原节点和复制节点的映射关系。第二次遍历链表，将复制节点的结构连接好。

```java
class Solution {
    public Node copyRandomList(Node head) {
        // 原节点-拷贝节点的映射
        Map<Node, Node> originToClone = new HashMap<>();
        // 第一次遍历，先将所有节点克隆出来
        for(Node p = head;p != null;p = p.next){
            originToClone.put(p, new Node(p.val));
        }
        // 第二次遍历，将克隆节点的结构连接好
        for(Node p = head;p != null;p = p.next){
            if(p.next != null){
                originToClone.get(p).next = originToClone.get(p.next);
            }
            if(p.random != null){
                originToClone.get(p).random = originToClone.get(p.random);
            }
        }
        // 返回克隆链表的头节点
        return originToClone.get(head);
    }
}
```

[148. 排序链表 - 力扣（LeetCode）](https://leetcode.cn/problems/sort-list/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：

先将所有节点添加到列表中，将节点按照值升序排序，再重建链表即可。

```java
class Solution {
    public ListNode sortList(ListNode head) {
        // 将所有节点添加到列表中，并升序排序
        List<ListNode> nodes = new ArrayList<>();
        for(ListNode p = head;p != null;p = p.next){
            nodes.add(p);
        }
        Collections.sort(nodes, (a, b) -> a.val - b.val);
        // 重建链表
        ListNode dummy = new ListNode(), p = dummy;
        for(int i = 0;i<nodes.size();i++){
            ListNode node = nodes.get(i);
            p.next = node;
            p = p.next;
        }
        // 将最后一个节点的指针置空
        p.next = null;
        return dummy.next;
    }
}
```

思路二：

使用归并排序方法，先使用快慢指针技巧找到链表的中间位置，再分别对子链表排序，再将子链表合并成升序链表即可。

```java
class Solution {
    public ListNode sortList(ListNode head) {
        return mergeSort(head, null);
    }

    private ListNode mergeSort(ListNode head, ListNode tail){
        // 空链表和只有一个元素的链表直接返回
        if(head == null) return null;
        if(head.next == tail){
            // 将节点的next指针断开
            head.next = null;
            return head;
        }
        // 获取链表中间位置的节点
        ListNode slow = head, fast = head;
        while(fast != tail && fast.next != tail){
            slow = slow.next;
            fast = fast.next.next;
        }
        // 递归调用，对子链表排序
        ListNode list1 = mergeSort(head, slow);
        ListNode list2 = mergeSort(slow, tail);
        // 将两条子链表合并成有序链表
        return merge(list1, list2);
    }

    ListNode merge(ListNode head1, ListNode head2){
        ListNode p1 = head1, p2 = head2;
        ListNode dummy = new ListNode(0), p = dummy;
        while(p1 != null && p2 != null){
            if(p1.val <= p2.val){
                p.next = p1;
                p1 = p1.next;
            }else{
                p.next = p2;
                p2 = p2.next;
            }
            p = p.next;
        }
        if(p1 != null){
            p.next = p1;
        }
        if(p2 != null){
            p.next = p2;
        }
        return dummy.next;
    }
}
```

[23. 合并 K 个升序链表 - 力扣（LeetCode）](https://leetcode.cn/problems/merge-k-sorted-lists/?envType=study-plan-v2&envId=top-100-liked)

思路：

由于链表已经是升序的，因此当前每条链表的头节点中值最小的即为当前值最小的节点，需要连接到结果链表中。可以使用最小堆保存节点，这样堆顶的即为最小的节点。先将每个链表的头节点添加到堆中，每次从堆顶取出当前最小的节点 node，连接到合并链表中，并将该节点的下一个节点添加到队中，直到队列为空结束。

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        // 优先级队列，最小堆
        PriorityQueue<ListNode> pq = new PriorityQueue<>((a,b) -> a.val - b.val);
        // 将 k 个链表的头节点加入最小堆
        for(ListNode node:lists){
            if(node != null){
                pq.add(node);
            }
        }
        ListNode dummy = new ListNode(0), p = dummy;

        while(!pq.isEmpty()){
            // 获取最小节点，接到结果链表中
            ListNode node = pq.poll();
            // 将节点的下一个节点添加到最小堆中
            if(node.next != null){
                pq.add(node.next);
            }
            p.next = node;
            p = p.next;
        }
        return dummy.next;
    }
}
```

[146. LRU 缓存 - 力扣（LeetCode）](https://leetcode.cn/problems/lru-cache/?envType=study-plan-v2&envId=top-100-liked)

思路：

为了保存和获取数据，可以使用哈希表，还需要维护数据是否为最近使用的，每次读取数据需要将数据设为最新的，当添加数据时，若容量已满，需要淘汰最久没有被使用过的，即需要维护数据的顺序，可以使用链表。Map集合中恰好有一个类 LinkedHashMap 既可以保存数据，也可以保证有序性，当获取元素时，将该元素移到链表尾部；当添加元素时，若容量已满，将链表头部的元素删除。

```java
class LRUCache {
    // 缓存
    LinkedHashMap<Integer,Integer> cache=new LinkedHashMap<>();
    int capacity;
    
    public LRUCache(int capacity) {
        this.capacity=capacity;
    }
    
    public int get(int key) {
        if(!cache.containsKey(key)) return -1;
        // 将数据移到队头并返回
        makeRecently(key);
        return cache.get(key);
    }
    
    public void put(int key, int value) {
        // 若缓存中存在，更新值并移到队尾
        if(cache.containsKey(key)){
            cache.put(key,value);
            makeRecently(key);
            return;
        }
        // 否则需要判断大小是否达到容量
        if(cache.size()==capacity){
            int oldestKey=cache.keySet().iterator().next();
            cache.remove(oldestKey);
        }
        cache.put(key,value);
    }
    
    void makeRecently(int key){
        int value=cache.get(key);
        // 删除key，重新插入到队尾
        cache.remove(key);
        cache.put(key,value);
    }
}
```

## 二叉树

[104. 二叉树的最大深度 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-depth-of-binary-tree/?envType=study-plan-v2&envId=top-100-liked)

思路一：

利用二叉树，使用变量 depth 记录当前遍历节点的深度，当到达叶子节点时更新最大深度即可。

```java
class Solution {
    // 记录遍历到的节点的深度
    int depth=0;
    // 记录最大深度
    int res=0;

    void traverse(TreeNode root){
        if(root==null) return;
        // 前序位置
        depth++;
        if(root.left==null&&root.right==null){
            //到达叶子节点，更新最大深度
            res=Math.max(res,depth);
        }
        traverse(root.left);
        traverse(root.right);
        // 后序位置
        depth--;
    }
    //定义：输入根节点，返回这颗二叉树的最大深度
    public int maxDepth(TreeNode root) {
        traverse(root);
        return res;
    }
}
```

思路二：

设$maxDepth_{left}$ 和 $maxDepth_{right}$为左右子树的最大深度，则以 root 为根节点的二叉树的最大深度为 $max(maxDepth_{left},maxDepth_{right}) + 1$。

```java
class Solution {
    //定义：输入根节点，返回这颗二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if(root == null) return 0;
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }
}
```

[226. 翻转二叉树 - 力扣（LeetCode）](https://leetcode.cn/problems/invert-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        traverse(root);
        return root;
    }
    void traverse(TreeNode root){
        if(root==null) return;
        traverse(root.left);
        traverse(root.right);
        // 交换左右子节点
        TreeNode tmp=root.left;
        root.left=root.right;
        root.right=tmp;
    }
}
```

[101. 对称二叉树 - 力扣（LeetCode）](https://leetcode.cn/problems/symmetric-tree/?envType=study-plan-v2&envId=top-100-liked)

思路一：

定义函数 traverse(root1, root2) 判断以root1和root2为根节点的两课子树是否是互相对称的，首先判断根节点是否相等，再递归调用函数，判断子节点是否对称。

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return traverse(root.left, root.right);
    }
    //定义：判断以root1和root2为根节点的两课子树是否是互相对称的
    boolean traverse(TreeNode root1, TreeNode root2){
        if(root1 == null && root2 == null) return true;
        if(root1 == null || root2 == null) return false;
        return root1.val == root2.val 
            && traverse(root1.left, root2.right) 
            && traverse(root1.right, root2.left);
    }
}
```

思路二：

迭代遍历二叉树，使用队列q保存遍历的节点，初始将根节点的左右子节点加入队列，每次从队列取出两个节点，判断节点是否相等，若不相等直接返回false，若相等则分别将两个节点的子节点按相反的顺序加入队列，使得对称的节点是相邻的，继续判断子节点是否对称，值到队列为空结束。

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return traverse(root.left, root.right);
    }
    boolean traverse(TreeNode u, TreeNode v){
        Queue<TreeNode> q = new LinkedList<>();
        // 初始将左右节点加入
        q.offer(u);
        q.offer(v);

        while(!q.isEmpty()){
            // 取出左右子树对称位置的节点并比较
            u = q.poll();
            v = q.poll();
            if(u == null && v == null) continue;
            if(u == null || v == null || u.val != v.val) return false;
            // 将两个节点的子节点分别加入（对称位置的放在一起）
            q.offer(u.left);
            q.offer(v.right);
            q.offer(u.right);
            q.offer(v.left);
        }
        return true;
    }
}
```

[543. 二叉树的直径 - 力扣（LeetCode）](https://leetcode.cn/problems/diameter-of-binary-tree/?envType=study-plan-v2&envId=top-100-liked)

思路：

直径即为树中最长路径的长度，经过节点node的最长路径长度为maxLeftDepth + maxRightDepth + 1，因此可以遍历二叉树，计算经过每个节点的最长路径长度，取最大值即为直径。同时，可以在计算maxDepth的同时，更新最长直径maxDeameter。

```java
class Solution {
    int maxDiameter = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        maxDepth(root);
        return maxDiameter;
    }
    int maxDepth(TreeNode root){
        if(root == null) return 0;
        int leftDepth = maxDepth(root.left);
        int rightDepth = maxDepth(root.right);
        maxDiameter = Math.max(maxDiameter, leftDepth + rightDepth);
        return Math.max(leftDepth, rightDepth) + 1;
    }
}
```

[102. 二叉树的层序遍历 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-tree-level-order-traversal/?envType=study-plan-v2&envId=top-100-liked)

思路：

使用队列q保存每一层的节点，初始将根节点加入队列，每次将上一层的节点弹出，并将弹出节点的子节点加入队列。

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> q = new LinkedList<>();
        if(root == null) return res;
        // 加入根节点
        q.offer(root);

        while(!q.isEmpty()){
            List<Integer> list = new ArrayList<>();
            int sz = q.size();
            // 将当前层的节点加入列表中
            for(int i=0;i < sz;i++){
                TreeNode node = q.poll();
                list.add(node.val);
                // 将弹出节点的子节点加入队列
                if(node.left != null){
                    q.offer(node.left);
                }
                if(node.right != null){
                    q.offer(node.right);
                }
            }
            res.add(list);
        }
        return res;
    }
}
```

[108. 将有序数组转换为二叉搜索树 - 力扣（LeetCode）](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        return traverse(nums, 0, nums.length - 1);
    }
    // 将区间[lo, hi]的元素转换为二叉搜索树
    public TreeNode traverse(int[] nums, int lo, int hi){
        if(lo > hi) return null;
        int mid = (lo + hi) >> 1;
        // 分别将中间位置左右两边的元素转换为二叉搜索树
        TreeNode leftChild = traverse(nums, lo, mid-1);
        TreeNode rightChild = traverse(nums, mid+1, hi);
        return new TreeNode(nums[mid], leftChild, rightChild);
    }
}
```

[98. 验证二叉搜索树 - 力扣（LeetCode）](https://leetcode.cn/problems/validate-binary-search-tree/?envType=study-plan-v2&envId=top-100-liked)

思路：

二叉搜索树需要满足左子节点的值小于父节点，右子节点的值大于父节点，可能我们会直接遍历二叉树，判断每个节点的左右子节点是否满足要求，但是这样判断不够的，对于每个节点，他的左子树的所有节点的值应该都小于该节点，右子树的所有节点的值应该都大于该节点。我们可以使用变量min, max限定当前节点值需要在 (min, max) 范围内，当遍历左子树时，限定max = root.val，当遍历右子树时，限定min = roo.val。

```java
class Solution {
    public boolean isValidBST(TreeNode root) {
        return isValid(root, null, null);
    }
    // 限定以 root 为根的节点必须满足 max.val > root.val > min.val
    public boolean isValid(TreeNode root, TreeNode min, TreeNode max){
        if(root == null) return true;
        if(min != null && root.val <= min.val) return false;
        if(max != null && root.val >= max.val) return false;
        // 限定左子树的最大值为 root.val，右子树的最小值为 root.val
        return isValid(root.left, min, root) && isValid(root.right, root, max);
    }
}
```

[230. 二叉搜索树中第 K 小的元素 - 力扣（LeetCode）](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int kthSmallest(TreeNode root, int k) {
        traverse(root,k);
        return res;
    }
    //结果
    int res;
    // 当前节点的排名
    int rank=0;
    void traverse(TreeNode root,int k){
        if(root==null) return;
        traverse(root.left,k);
        // 中序代码位置
        rank++;
        if(rank==k){
            // 找到第 k 小的元素
            res=root.val;
            return;
        }
        traverse(root.right,k);
    }
}
```

[199. 二叉树的右视图 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-tree-right-side-view/?envType=study-plan-v2&envId=top-100-liked)

思路：

二叉树的右视图即树的最右边的节点的列表，可以对二叉树进行层序遍历，每一层的最后一个元素即为结果。

```java
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Queue<TreeNode> q = new LinkedList<>();
        if(root == null) return res;
        q.offer(root);
        while(!q.isEmpty()){
            int sz = q.size();
            for(int i=0;i<sz;i++){
                TreeNode node = q.poll();
                // 将每一层的最后一个节点加入结果
                if(i == sz-1) res.add(node.val);
                if(node.left != null) {
                    q.offer(node.left);
                }
                if(node.right != null){
                    q.offer(node.right);
                }
            }
        }
        return res;
    }
}
```

[114. 二叉树展开为链表 - 力扣（LeetCode）](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/?envType=study-plan-v2&envId=top-100-liked)

思路：

先分别将左右子树展开为链表，再将左链表插入到右边即可，但是在连接连表时需要使用尾节点，因此在展开链表后需要返回链表的尾节点。

```java
class Solution {
    public void flatten(TreeNode root) {
        if(root == null) return;
        traverse(root);   
    }
    // 将根节点为 root 的二叉树展开为链表，并返回链表尾节点
    public TreeNode traverse(TreeNode root){
        if(root.left == null && root.right == null) return root;
        if(root.left == null){
            return traverse(root.right);
        }
        if(root.right == null){
            TreeNode leftTail = traverse(root.left);
            root.right = root.left;
            root.left = null;
            return leftTail;
        }
        // 分别将左右子树展开
        TreeNode leftTail = traverse(root.left);
        TreeNode rightTail = traverse(root.right);
        TreeNode rightHead = root.right;
        // 将左链表插入到右边
        root.right = root.left;
        root.left = null;
        leftTail.right = rightHead;
        
        // 返回链表的尾节点
        return rightTail;
    }
}
```

[105. 从前序与中序遍历序列构造二叉树 - 力扣（LeetCode）](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/?envType=study-plan-v2&envId=top-100-liked)

思路：

前序遍历数组的第一个元素即为根节点的值rootValue，设其在中序遍历数组中的位置为index，则左子树的节点数为 index-in_lo，构建根节点，并递归调用函数，构建左右子树即可。

```java
class Solution {
    // 存储 inorder 中值到索引的映射
    Map<Integer, Integer> numToIdx = new HashMap<>();

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        for(int i=0;i<inorder.length;i++){
            numToIdx.put(inorder[i], i);
        }
        return build(preorder, inorder, 0, preorder.length - 1, 0, inorder.length - 1);
    }

    // 定义：前序遍历数组为 preorder[pre_lo..pre_hi], 中序遍历数组为 inorder[in_lo..inhi]
    // 构造这个二叉树并返回根节点
    public TreeNode build(int[] preorder, int[] inorder, int pre_lo, int pre_hi, int in_lo, int in_hi){
        if(pre_lo > pre_hi) return null;

        // root节点的值就为前序遍历数组的第一个元素
        int rootValue = preorder[pre_lo];
        // rootValue 在中序遍历数组中的位置
        int index = numToIdx.get(rootValue);
        // 左子树的节点数
        int left_len = index - in_lo;
        // 构造根节点
        TreeNode root = new TreeNode(rootValue);
        // 递归构造左右子树
        root.left = build(preorder, inorder, pre_lo + 1, pre_lo + left_len, in_lo, index - 1);
        root.right = build(preorder, inorder, pre_lo + left_len + 1, pre_hi, index + 1, in_hi);
        // 返回根节点
        return root;
    }
}
```

[437. 路径总和 III - 力扣（LeetCode）](https://leetcode.cn/problems/path-sum-iii/?envType=study-plan-v2&envId=top-100-liked)

思路一：

定义函数 rootSum(root, target) 计算以 root 为起点且满足路径总和为 targetSum 的路径数目，再遍历二叉树，分别计算以每个节点为起点的满足要求的路径数目，累加结果即可。

```java
class Solution {
    public int pathSum(TreeNode root, int targetSum) {
        if(root == null) return 0;
        // 分别计算以每个节点为起点的满足要求的路径数目，并累加结果
        return rootSum(root, targetSum) + pathSum(root.left, targetSum) + pathSum(root.right, targetSum);
    }
    // 以 root 为起点且满足路径总和为 targetSum 的路径数目
    public int rootSum(TreeNode root, long targetSum){
        if(root == null) return 0;
        int res = 0;
        if(root.val == targetSum) res++;
        // 递归计算左右子树满足要求的路径数目
        res += rootSum(root.left, targetSum - root.val) + rootSum(root.right, targetSum - root.val);
        return res;
    }
}
```

思路二：

定义节点的前缀和为根节点到当前节点间所有节点的和，使用 prefix 保存当前已经遍历节点的前缀和及出现次数，若当前节点的前缀和为 cur，若已遍历节点中存在前缀和为 cur- targetSum 的，则该节点到当前节点的路径上所有节点的和为 targetSum，即满足要求的路径数为 prefix[cur- targetSum]。递归遍历子节点，计算所有满足要求的路径，同时在退出当前节点时需要更新 prefix。

```java
class Solution {
    // 当前已遍历路径的前缀和及数量
    Map<Long, Integer> prefix = new HashMap<>();
    public int pathSum(TreeNode root, int targetSum) {
        // 初始化路径长度为0的路径数量为0
        prefix.put(0L, 1);
        return traverse(root, 0, targetSum);
    }

    // 定义：返回以 root 为根节点的树中路径长度为 targetSum 的路径总数，curr 为当前经过的路径长度
    int traverse(TreeNode root, long curr, int targetSum){
        if(root == null) return 0;
        curr += root.val;
        // 路径前缀和为 curr - targetSum 的路径数目
        int res = prefix.getOrDefault(curr - targetSum, 0);
        // 将当前路径加入 prefix
        prefix.put(curr, prefix.getOrDefault(curr, 0) + 1);
        // 递归调用函数，计算左右子树满足要求的路径数目
        res += traverse(root.left, curr, targetSum) + traverse(root.right, curr, targetSum);
        // 回溯（更新路径前缀和）
        prefix.put(curr, prefix.get(curr) - 1);
        return res;
    }
}
```

[236. 二叉树的最近公共祖先 - 力扣（LeetCode）](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/?envType=study-plan-v2&envId=top-100-liked)

思路：

根据函数定义，lowestCommonAncestor(root, p, q)计算节点p, q在以root为根节点的树中的最近公共祖先，递归调用函数，传入左子节点left和右子节点right，返回left和right，可以分为三种情况：

- p, q分别在左右子树中(left != null && right != null)，则根节点即为节点p, q的最近公共祖先
- p, q都不在左右子树中(left == null && right == null)，即树中不存在节点p, q的祖先，返回null
- p和q同时在左子树中或同时在右子树中(left == null || right == null)，则说明最近公共祖先节点在左子树或右子树中，返回对应祖先节点即可

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // base case
        if(root==null) return null;
        if(root == p || root == q) return root;

        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if(left != null && right != null) return root;
        if(left == null && right == null) return null;
        return left == null ? right:left;
    }
}
```

[124. 二叉树中的最大路径和 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-tree-maximum-path-sum/?envType=study-plan-v2&envId=top-100-liked)

思路：

定义节点的贡献值为以该节点为起点的路径上所有的节点值之和，定义maxGain(root)计算root节点的最大贡献值，则当前节点的最大贡献值等于左右子节点的贡献值的较大值加上当前节点值，即max(maxLeft, maxRight) + root.val，则经过当前节点的最大路径和为maxLeft+maxRight+root.val。

```java
class Solution {
    int maxPath = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        maxGain(root);
        return maxPath;
    }

    // 定义：返回节点的最大贡献值
    // 贡献值：在以该节点为根节点的子树中，以该节点为起点的路径上所有节点值之和
    public int maxGain(TreeNode root){
        if(root == null) return 0;

        // 递归计算左右子节点的最大贡献值
        // 只有在最大贡献值大于 0 时，才会选取对应子节点
        int maxLeft = Math.max(0, maxDepth(root.left));
        int maxRight = Math.max(0, maxDepth(root.right));
        
        // 节点的最大路径和为该节点的值与左右子节点最大贡献值的和
        maxPath = Math.max(maxPath, maxLeft + maxRight + root.val);

        // 返回节点的最大贡献值
        return Math.max(maxLeft, maxRight) + root.val;
    }
}
```

## 图论

[200. 岛屿数量 - 力扣（LeetCode）](https://leetcode.cn/problems/number-of-islands/?envType=study-plan-v2&envId=top-100-liked)

思路一：

使用dfs搜索方法，每次从一个陆地节点开始搜索，并递归搜索四周节点，每完成一次搜索就遍历完了一个连通分支(岛屿)，结果加一，值到陆地都被遍历完结束，同时还需要定义visited记录已访问过的位置，避免走回头路。

```java
class Solution {
    int n,m;
    int[][] d = {{-1,0},{0,1},{1,0},{0,-1}};
    // 是否访问过
    boolean[][] visited;
    public int numIslands(char[][] grid) {
        n = grid.length; 
        m = grid[0].length;
        visited = new boolean[n][m];
        int res = 0;
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                // 只有陆地才遍历
                if(grid[i][j]=='1'){
                    res++;
                    dfs(i,j,grid);
                }
            }
        }
        return res;
    }
    void dfs(int i, int j, char[][] grid){
        // 超出边界，返回
        if(i<0 || i>n-1 || j<0 || j>m-1) return;
        // 跳过海水和已访问位置
        if(grid[i][j] == '0' || visited[i][j]) return;
        // 标记该位置已被访问
        visited[i][j] = true;
        // 遍历四周的位置
        for(int k=0;k<4;k++){
            dfs(i + d[k][0], j + d[k][1], grid);
        }
    }
}
```

思路二：

不使用visited数组，而是在每遍历一个位置时，就将该位置设置为海水位置，遍历时跳过海水位置。

```java
class Solution {
    int n,m;
    int[][] d={{-1,0},{0,1},{1,0},{0,-1}};
    public int numIslands(char[][] grid) {
        n = grid.length; 
        m = grid[0].length;
        int res=0;
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                // 只有陆地才遍历
                if(grid[i][j]=='1'){ 
                    res++;
                    dfs(i,j,grid);
                }
            }
        }
        return res;
    }
    void dfs(int i, int j, char[][] grid){
        // 超出边界，返回
        if(i<0 || i>n-1 || j<0 || j>m-1) return;
        // 已经是海水了
        if(grid[i][j] == '0') return;
        // 将(i,j)变成海水
        grid[i][j]='0';
        for(int k=0;k<4;k++){
            dfs(i + d[k][0], j + d[k][1], grid);
        }
    }
}
```

[207. 课程表 - 力扣（LeetCode）](https://leetcode.cn/problems/course-schedule/?envType=study-plan-v2&envId=top-100-liked)

思路一：

将课程抽象成一个有向图，节点数为课程总数，对于课程前置要求(u, v)，添加一条v指向u的边。使用拓扑排序方法，使用indegree保存每个节点的入度，初始将入度为0的节点加入队列，遍历队列中的节点u，并更新相邻节点的入度indegree[v]--，若入度变为0，则将该节点v加入队列，记录遍历课程的数量(可完成的课程数量)，若等于课程总数，则说明可以完成所有课程。

```java
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        // 记录每个节点的入度
        int[] indegree = new int[numCourses];
        List<Integer>[] graph = new ArrayList[numCourses];
        for(int i=0;i<numCourses;i++) graph[i] = new ArrayList<>();
        for(int[] p:prerequisites){
            int a = p[0], b = p[1];
            graph[b].add(a);
            indegree[a]++;
        }
        // 保存入度为0的节点
        Queue<Integer> q=new LinkedList<>();
        for(int i=0;i<numCourses;i++){
            if(indegree[i]==0) q.offer(i);
        }
        // 当前可以修的课程数量
        int count = 0;
        while(!q.isEmpty()){
            // 取出一个入度为0的节点
            int x = q.poll();
            count++;
            // 将邻接节点入度-1
            for(int y:graph[x]){
                indegree[y]--;
                if(indegree[y] == 0) q.offer(y);
            }
        }
        // 当可以修的课程数等于课程总数时返回true
        return count == numCourses;
    }
}
```

思路二：

使用深度优先搜索方法，定义visited记录遍历过的节点，onPath记录当前遍历的路径上的节点，当遍历的某个节点已经存在于onPath中时，说明存在环，无法完成所有课程。

```java
class Solution {
    // 记录遍历过的节点
    boolean[] visited;
    // 记录遍历过程中路径上的节点
    boolean[] onPath;
    // 记录图中是否有环
    boolean hasCycle;
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<Integer>[] g = buildGraph(numCourses, prerequisites);
        visited = new boolean[numCourses];
        onPath = new boolean[numCourses];
        // 遍历图中的所有节点
        for(int i = 0;i<numCourses;i++){
            dfs(g, i);
        }
        // 当没有环时可以完成所有课程
        return !hasCycle;
    }

    void dfs(List<Integer>[] g, int x){
        if(onPath[x]) {
            // 出现环
            hasCycle = true;
        }
        // 如果出现环或已经访问过该节点，返回
        if(hasCycle || visited[x]) return;
        // 前序代码位置
        visited[x] = true;
        onPath[x] = true;
        for(int y:g[x]){
            dfs(g, y);
        }
        // 后序代码位置
        onPath[x] = false;
    }

    List<Integer>[] buildGraph(int numCourses, int[][] prerequisites){
        List<Integer>[] g = new ArrayList[numCourses];
        for(int i=0;i<numCourses;i++) g[i] = new ArrayList<>();
        for(int[] edge:prerequisites){
            // 修完课程 v 才能修课程 u
            // 在图中添加一条从 v 指向 u 的有向边
            int u = edge[0], v = edge[1];
            g[v].add(u);
        }
        return g;
    }
}
```

[994. 腐烂的橘子 - 力扣（LeetCode）](https://leetcode.cn/problems/rotting-oranges/?envType=study-plan-v2&envId=top-100-liked)

思路一：

定义 $dist[i][j]$记录位置 (i, j) 的新鲜橘子腐烂需要经过的最小分钟数，初始化为极大值，以每个腐烂橘子为起点进行bfs搜索，在遍历时更新dist数组，将新鲜橘子变成腐烂橘子状态，最后计算每个新鲜橘子需要经过最小分钟数的最大值即为结果，同时还需要判断是否存在新鲜橘子，若存在，说明无法使所有橘子腐烂，返回-1。

```java
class Solution {
    int res = 0;
    int m, n;
    int[][] d = {{1,0},{-1,0},{0,1},{0,-1}};
    // 记录每个新鲜橘子腐烂必须经过的最小分钟数
    int[][] dist;
    public int orangesRotting(int[][] grid) {
        m = grid.length;
        n = grid[0].length;
        dist = new int[m][n];
        // 初始化dist
        for(int i=0;i < m;i++){
            for(int j=0;j < n;j++){
                if(grid[i][j] == 1) dist[i][j] = Integer.MAX_VALUE;
            }
        }
        // 分别以每一个腐烂橘子为起点，进行bfs搜索
        for(int i = 0;i < m;i++){
            for(int j = 0;j < n;j++){
                if(grid[i][j] == 2) {
                    bfs(grid, i, j);
                }
            }
        }
        // 所有新鲜橘子腐烂需要分钟数的最大值即为结果
        int res = 0;
        for(int i = 0;i < m;i++){
            for(int j = 0;j < n;j++){
                if(grid[i][j] == 1){
                    res = Math.max(res, dist[i][j]);
                }
            }
        }

        return res == Integer.MAX_VALUE ? -1:res;
    }

    void bfs(int[][] grid, int u, int v){
        // 记录已经访问过的位置
        boolean[][] visited = new boolean[m][n];
        // 记录当前搜索的层数
        int rank = 0;
        // 创建节点队列，并加入起始节点
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[]{u, v});
        while(!q.isEmpty()){
            int sz = q.size();
            while(sz-- > 0){
                int[] node = q.poll();
                int x = node[0], y = node[1];
                if(visited[x][y]) continue;
                visited[x][y] = true;
                // 更新当前位置橘子腐烂需要的分钟数
                dist[x][y] = Math.min(dist[x][y], rank);
                // 将相邻节点加入队列
                for(int k=0;k<4;k++){
                    int nx = node[0] + d[k][0], ny = node[1] + d[k][1];
                    if(nx < 0 || nx > m-1 || ny < 0 || ny > n-1) continue;
                    if(grid[nx][ny] == 1) q.offer(new int[]{nx, ny});
                }
            }
            rank++;
        }
    }
}
```

思路二：

使用多源广度优先搜索方法，初始将所有腐烂橘子节点加入队列，并开始进行bfs搜索，在搜索时每一层弹出队列中的节点，并将相邻的新鲜橘子节点加入队列，将其设置为腐烂橘子，最终遍历的最大层数即为结果，遍历完还需要判断是否存在新鲜橘子节点，若存在说明无法使所有橘子腐烂，返回-1。

```java
class Solution {
    int m, n;
    int[][] d = {{1,0},{-1,0},{0,1},{0,-1}};
    public int orangesRotting(int[][] grid) {
        m = grid.length;
        n = grid[0].length;
        int res = bfs(grid);
        // 如果搜索完后还存在新鲜的橘子，直接返回
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j] == 1) return -1;
            }
        }
        return res;
    }

    int bfs(int[][] grid){
        // 记录当前搜索的层数
        int rank = 0;
        // 创建节点队列，并加入初始节点（腐烂橘子节点）
        Queue<int[]> q = new LinkedList<>();
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j] == 2) q.offer(new int[]{i, j});
            }
        }
        // 单元格中没有新鲜橘子需要经过的最小分钟数
        int res = 0;
        while(!q.isEmpty()){
            int sz = q.size();
            while(sz-- > 0){
                int[] node = q.poll();
                int x = node[0], y = node[1];
                res = Math.max(res, rank);
                // 将相邻节点加入队列
                for(int k=0;k<4;k++){
                    int nx = node[0] + d[k][0], ny = node[1] + d[k][1];
                    if(nx < 0 || nx > m-1 || ny < 0 || ny > n-1) continue;
                    // 将新鲜橘子节点加入队列
                    if(grid[nx][ny] == 1){
                        q.offer(new int[]{nx, ny});
                        // 将新鲜橘子节点标记为腐烂
                        grid[nx][ny] = 2;
                    } 
                }
            }
            rank++;
        }
        return res;
    }
}
```

[208. 实现 Trie (前缀树) - 力扣（LeetCode）](https://leetcode.cn/problems/implement-trie-prefix-tree/description/?envType=study-plan-v2&envId=top-100-liked)

```java
class Trie {
    TrieSet set=new TrieSet();
    public Trie() {
    }
    
    public void insert(String word) {
        set.add(word);
    }
    
    public boolean search(String word) {
        return set.contains(word);
    }
    
    public boolean startsWith(String prefix) {
        return set.hasKeyWithPrefix(prefix);
    }
}
class TrieSet{
    TrieMap<Object> map = new TrieMap<>();

    void add(String key){
        map.put(key,new Object());
    }

    void remove(String key){
        map.remove(key);
    }

    boolean contains(String key){
        return map.containsKey(key);
    }

    String shortestPrefixOf(String query){
        return map.shortestPrefixOf(query);
    }

    String longestPrefixOf(String query){
        return map.longestPrefixOf(query);
    }

    List<String> keysWithPrefix(String prefix){
        return map.keysWithPrefix(prefix);
    }

    boolean hasKeyWithPrefix(String prefix){
        return map.hasKeyWithPrefix(prefix);
    }

    List<String> keysWithPattern(String pattern){
        return map.keysWithPattern(pattern);
    }

    boolean hasKeyWithPattern(String pattern){
        return map.hasKeyWithPattern(pattern);
    }
}
class TrieMap<V>{
    // 字母的个数
    static final int R = 26;
    // 元素个数
    int size = 0;
    // 字典树根节点
    TrieNode<V> root = new TrieNode<>();

    /* 字典树节点的结构 */
    static class TrieNode<V>{
        V val=null;
        TrieNode<V>[] children = new TrieNode[R];
    }

    // 添加或更新一个元素
    void put(String key,V val){
        if(!containsKey(key)) size++;
        put(root, key, val, 0);
    }

    // 定义：向以node为根节点的trie树中插入key[i..]，返回插入完成后的根节点
    TrieNode<V> put(TrieNode<V> node, String key, V val, int i){
        if(node == null){
            // 如果树枝不存在，新建
            node = new TrieNode<>();
        }
        // System.out.println(node);
        if(i == key.length()) {
            // key的路径已经插入完成，将值val存入节点
            node.val = val;
            return node;
        }
        int idx = key.charAt(i) - 'a';
        // 递归插入子节点，并接收返回值
        node.children[idx] = put(node.children[idx], key, val, i+1);
        return node;
    }

    // 删除一个元素
    void remove(String key){
        if(!containsKey(key)) return;
        remove(root,key,0);
        size--;
    }

    // 定义：在以node为根的Trie树种删除key[i..]，返回删除后的根节点
    TrieNode<V> remove(TrieNode<V> node, String key, int i){
        if(i == key.length()) {
            // 找到key对应的TrieNode，删除val
            node.val = null;
        }else{
            int idx = key.charAt(i)-'a';
            // 递归去子树删除
            node.children[idx] = remove(node.children[idx], key, i+1);
        }
        
        /*后序位置，递归路径上的节点可能需要被清理*/
        // 如果该节点存储着val，不需要清理
        if(node.val != null) return node;
        // 检查该节点是否还有后缀
        for(int j = 0;j < R;j++){
            // 只要存在一个子节点，就不需要清理
            if(node.children[j] != null) return node;
        }

        // 没有存储val，也没有后缀数枝，则该节点需要被清理
        return null;
    }

    // 搜索key对应的值，不存在返回null
    V get(String key){
        TrieNode<V> node = getNode(root, key);
        if(node == null || node.val == null) return null;
        return node.val;
    }

    // 在以 node 为根节点的树中查找key对应的节点
    TrieNode<V> getNode(TrieNode<V> node,String key){
        TrieNode<V> p = node;
        for(int i = 0;i < key.length();i++){
            if(p == null) return null;
            int idx = key.charAt(i) - 'a';
            p = p.children[idx];
        }
        return p;
    }

    // 判断key是否存在map中
    boolean containsKey(String key){
        return get(key) != null;
    }

    // 判断是否存在前缀为prefix的键
    boolean hasKeyWithPrefix(String prefix){
        return getNode(root, prefix) != null;
    }

    // 在所有键中寻找query的最短前缀
    String shortestPrefixOf(String query){
        TrieNode<V> p = root;
        for(int i = 0;i < query.length();i++){
            // 无法向下搜索
            if(p == null) return "";
            // 找到一个键是query的前缀
            if(p.val != null) return query.substring(0,i);
            // 向下搜索
            int idx = query.charAt(i) - 'a';
            p = p.children[idx];
        }
        // 如果query本身就是一个键
        if(p != null && p.val != null) return query;
        return "";
    }

    // 在所有键中寻找query的最长前缀
    String longestPrefixOf(String query){
        TrieNode<V> p = root;
        // 记录前缀的最大长度
        int max_len = 0;
        for(int i = 0;i < query.length();i++){
            // 无法向下搜索
            if(p == null) return "";
            // 更新前缀的长度
            if(p.val != null) max_len = i;
            int idx = query.charAt(i)-'a';
            p = p.children[idx];
        }
        // query本身是一个键
        if(p != null && p.val != null) return query;
        return query.substring(0, max_len);
    }

    // 搜索前缀为prefix的所有键
    List<String> keysWithPrefix(String prefix){
        List<String> res = new ArrayList<>();
        // 找到匹配prefix在树中的对应节点
        TrieNode<V> x = getNode(root, prefix);
        // 不存在匹配的，返回
        if(x == null) return res;
        // dfs遍历以 x 为根节点的子树
        traverse(x, new StringBuilder(prefix), res);
        return res;
    }

    // 遍历已node节点为根的树，找到所有键
    void traverse(TrieNode<V> node, StringBuilder path, List<String> res){
        if(node == null) return;
        if(node.val != null) res.add(path.toString());
        for(int i = 0;i < R;i++){
            if(node.children[i] != null){
                // 做选择
                path.append('a' + i);
                traverse(node.children[i], path, res);
                // 撤销选择
                path.deleteCharAt(path.length() - 1);
            }
        }
    }

    // 搜索所有匹配pattern通配符的键
    List<String> keysWithPattern(String pattern){
        List<String> res = new ArrayList<>();
        traverse(root, new StringBuilder(), pattern, 0, res);
        return res;
    }

    // 从node节点开始搜索匹配pattern[i..]的键，path记录搜索路径
    void traverse(TrieNode<V> node, StringBuilder path, String pattern, int i, List<String> res){
        if(node == null) return;
        if(i == pattern.length()) {
            // 找到一个匹配的键
            if(node.val != null) res.add(path.toString());
            return;
        }
        char c = pattern.charAt(i);
        // 如果字符为通配符，可以匹配任意字符
        if(c == '.'){
            for(int j = 0;j < R;j++){
                if(node.children[j] != null){
                    path.append('a' + j);
                    traverse(node.children[j], path, pattern, i+1, res);
                    path.deleteCharAt(path.length() - 1);
                }
            }
        }else{
            path.append(c);
            traverse(node.children[c-'a'], path, pattern, i+1, res);
            path.deleteCharAt(path.length() - 1);
        }
    }

    // 判断是否存在匹配通配符pattern的键
    boolean hasKeyWithPattern(String pattern){
        return hasKeyWithPattern(root, pattern, 0);
    }

    // 已node节点为根节点，搜索是否存在匹配通配符pattern[i..]的键
    boolean hasKeyWithPattern(TrieNode<V> node, String pattern, int i){
        if(node == null) return false;
        if(i == pattern.length()) {
            return node.val != null;
        }
        char c = pattern.charAt(i);
        if(c=='.'){
            for(int j = 0;j < R;j++){
                if(node.children[j] != null){
                    return hasKeyWithPattern(node.children[j], pattern, i+1);
                }
            }
        }else{
            return hasKeyWithPattern(node.children[c-'a'], pattern, i+1);
        }
        return false;
    }
}
```

## 回溯

[46. 全排列 - 力扣（LeetCode）](https://leetcode.cn/problems/permutations/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    List<List<Integer>> res=new ArrayList<>();
    // 记录每个元素是否被使用
    boolean[] used;
    // 主函数，输入一组不重复的数字，返回它们的全排列
    public List<List<Integer>> permute(int[] nums) {
        int n = nums.length;
        used = new boolean[n];
        traverse(nums, new LinkedList<Integer>());
        return res;
    }

    // 路径：记录在track中
    // 选择列表：nums 中不存在于 track 的那些元素（used[i] 为false）
    // 结束条件：nums 中的元素全都在 track 中出现
    void traverse(int[] nums,LinkedList<Integer> track){
        // 到达叶子节点，添加结果
        if(track.size() == nums.length){
            res.add(new LinkedList(track));
            return;
        }
        // 枚举下一个选择
        for(int i = 0;i < nums.length;i++){
            if(used[i]) continue;
            // 做选择
            track.add(nums[i]);
            used[i] = true;
            // 进入下一层决策树
            traverse(nums,track);
            // 取消选择
            track.removeLast();
            used[i] = false;
        }
    }
}
```

[78. 子集 - 力扣（LeetCode）](https://leetcode.cn/problems/subsets/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    List<List<Integer>> res=new ArrayList<>();
    // 记录回溯算法递归路径
    List<Integer> track = new LinkedList<>();
    public List<List<Integer>> subsets(int[] nums) {
        traverse(nums,0);
        return res;
    }
    
    void traverse(int[] nums, int start){
        // 前序位置，每个节点的值都是一个子集
        res.add(new LinkedList(track));
        // 枚举下一个元素
        for(int i = start;i < nums.length;i++){
            track.add(nums[i]);
            traverse(nums, i+1);
            track.removeLast();
        }
    }
}
```

[17. 电话号码的字母组合 - 力扣（LeetCode）](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/?envType=study-plan-v2&envId=top-100-liked)

思路：

直接使用回溯算法，将每个数字对应的字母保存到mapping中，对于digits中的每一个数字，枚举对应的字母，添加到路径上，并进入下一层回溯树，当枚举完数字后，找到了一个结果，添加到结果列表中。

```java
class Solution {
    // 数字到字母的映射
    String[] mapping = new String[]{"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
    List<String> res = new ArrayList<>();

    // 回溯
    void backtrack(String digits, int i, StringBuilder sb){
        if(i == digits.length()){
            res.add(sb.toString());
            return;
        }
        int x = digits.charAt(i) - '0';
        // 枚举数字可以表示的每一个字母
        for(char c:mapping[x].toCharArray()){
            // 做选择
            sb.append(c);
            // 递归下一层回溯树
            dfs(digits, i+1, sb);
            // 撤销选择
            sb.deleteCharAt(sb.length() - 1);
        }
    }
    public List<String> letterCombinations(String digits) {
        if(digits.isEmpty()) return res;
        StringBuilder sb = new StringBuilder(); 
        dfs(digits,0,sb);
        return res;
    }
}
```

[39. 组合总和 - 力扣（LeetCode）](https://leetcode.cn/problems/combination-sum/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    // 记录回溯算法递归路径
    List<Integer> track = new LinkedList<>();
    //记录路径上元素的和
    int trackSum = 0;
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        traverse(candidates, 0, target);
        return res;
    }
    
    void traverse(int[] nums,int start,int target){
        // 前序位置，每个节点的值都是一个子集
        if(trackSum == target){
            res.add(new LinkedList(track));
            return;
        }
        // 超出目标值，剪枝
        if(trackSum > target) return;
        // 枚举下一个选择
        for(int i = start;i < nums.length;i++){
            track.add(nums[i]);
            trackSum += nums[i];
            // 元素可以重复选择
            traverse(nums, i, target);
            track.removeLast();
            trackSum -= nums[i];
        }
    }
}
```

[22. 括号生成 - 力扣（LeetCode）](https://leetcode.cn/problems/generate-parentheses/?envType=study-plan-v2&envId=top-100-liked)

思路：

- 使用回溯算法，使用track记录当前的括号串，在每一层可以选择一个左括号或右括号添加到括号串中
- 同时使用left, right记录当前剩余的左右括号数量，如果剩余左括号数量大于剩余右括号数量，或者括号数量小于0时，都是不合法的，直接返回；如果左右括号数都为0，则找到了一个合法结果

```java
class Solution {
    List<String> res=new ArrayList<>();
    public List<String> generateParenthesis(int n) {
        // 可用的左括号和右括号数量初始化为n
        traverse(n, n, new StringBuilder());
        return res;
    }

    // 可用的左括号数量和右括号数量分别为left, right
    void traverse(int left, int right, StringBuilder track){
        // 若左括号剩下的多，说明不合法
        if(left > right) return;
        // 数量小于 0 是不合法的
        if(left < 0 || right < 0) return;
        // 当所有括号都恰好用完时，得到一个合法的括号组合
        if(left == 0 && right == 0){
            res.add(track.toString());
            return;
        }
        
        // 尝试添加一个左括号
        // 选择
        track.append("(");
        traverse(left-1, right, track);
        // 撤销选择
        track.deleteCharAt(track.length() - 1);
        // 尝试添加一个右括号
        // 选择
        track.append(")");
        traverse(left, right-1, track);
        // 撤销选择
        track.deleteCharAt(track.length() - 1);
    }
}
```

[79. 单词搜索 - 力扣（LeetCode）](https://leetcode.cn/problems/word-search/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    int[][] d = {{1,0},{-1,0},{0,1},{0,-1}};
    // 标记是否访问过
    boolean[][] visited;
    int m, n;
    // 是否找到合法单元格
    boolean found = false;
    public boolean exist(char[][] board, String word) {
        m = board.length;
        n = board[0].length;
        visited = new boolean[m][n];
        for(int i = 0;i < m;i++){
            for(int j = 0;j < n;j++){
                dfs(board, i, j, word, 0);
                if(found) return true;
            }
        }
        return false;
    }

    // 从(i, j)开始向四周搜索，试图匹配 word[p..]
    void dfs(char[][] board, int x,int y, String word, int i){
        if(i == word.length()){
            // word被匹配完，找到一个答案
            found = true;
            return;
        }
        // 已经找到一个答案，不用搜索了
        if(found) return;
        if(x < 0 || x > m-1 || y < 0 || y > n-1) return;
        if(visited[x][y]) return;
        if(board[x][y] != word.charAt(i)) return;
        // 标记已被访问
        visited[x][y] = true;
        // word[i]已被匹配，开始向四周搜索 word[i+1..]
        for(int k = 0;k < 4;k++){
            int nx = x + d[k][0], ny = y + d[k][1];
            dfs(board, nx, ny, word, i+1);
        }
        // 撤销
        visited[x][y] = false;
    }
}
```

[131. 分割回文串 - 力扣（LeetCode）](https://leetcode.cn/problems/palindrome-partitioning/?envType=study-plan-v2&envId=top-100-liked)

思路：

- 使用回溯算法，使用i表示当前串 s[0..i-1] 已被分割为子串，需要将串 s[i..n-1] 分割成子串，则可以枚举下一个字串的结束位置j，当s[i..j] 是一个回文串时，则是合法的，开始分割下一个子串。
- 为了判断串 s[i..j] 是否为回文串，可以先预处理一下，定义 $f[i][j]$ 表示串s[i..j]是否为回文串，则$f[i][j]=f[i-1][j+1]\&\&s[i]=s[j](i<j),f[i][j]=true(i>=j)$，即只有一个字符的子串和空串为回文串，其它的当首尾字符相等且中间的子串为回文串时才为回文串

```java
class Solution {
    //f[i][j]: 子串s[i..j]是否为回文串
    boolean[][] f;
    // 结果
    List<List<String>> res = new ArrayList<>();
    // 路径（当前分割的子串列表）
    List<String> track = new LinkedList<>();
    public List<List<String>> partition(String s) {
        int n = s.length();
        f = new boolean[n][n];
        for(boolean[] t:f) Arrays.fill(t, true);
        for(int i = n-1;i>=0;i--){
            for(int j = i+1;j<n;j++){
                f[i][j] = (s.charAt(i) == s.charAt(j)) && f[i+1][j-1];
            }
        }
        backtrack(s, 0);
        return res;
    }

    // s[0..i-1]都已被分割成子串，需要将剩余的串s[i..n-1]分割成子串
    void backtrack(String s, int i){
        int n = s.length();
        // 字符串被分割完了，找到了一个结果
        if(i == n){
            res.add(new ArrayList<>(track));
            return;
        }
        // 枚举当前可以分割的回文串
        for(int j = i;j < n;j++){
            if(f[i][j]){
                track.add(s.substring(i, j+1));
                backtrack(s, j+1);
                track.removeLast();
            }
        }
    }
}
```

[51. N 皇后 - 力扣（LeetCode）](https://leetcode.cn/problems/n-queens/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    List<List<String>> res = new ArrayList<>();

    // 输入期盼边长 n，返回所有合法的放置
    public List<List<String>> solveNQueens(int n) {
        char[][] board = new char[n][n];
        // 初始化空棋盘
        for(int i=0;i<n;i++){
            for(int j = 0;j<n;j++) board[i][j] = '.';
        }
        backtrack(board, 0);
        return res;
    }

    // 路径：board中小于row的那些行都已经成功放置了皇后
    // 选择列表：第 row 行的所有列都是防止皇后的选择
    // 结束条件：row 超过 board 的最后一行
    void backtrack(char[][] board,int row){
        // 触发结束条件
        if(row == board.length){
            List<String> list = new ArrayList<>();
            for(char[] r:board){
                list.add(new String(r));
            }
            res.add(list);
            return;
        }

        int n = board.length;
        for(int col = 0;col < n;col++){
            // 排除不合法选择
            if(!isValid(board, row, col)) continue;
            // 做选择
            board[row][col] = 'Q';
            // 进入下一行决策
            backtrack(board, row + 1);
            // 撤销选择
            board[row][col] = '.';
        }
    }
    // 判断当前选择是否合法
    boolean isValid(char[][] board, int row, int col){
        // 判断列上是否重复
        for(int i=0;i<row;i++){
            if(board[i][col]=='Q') return false; 
        }
        // 判断左上方是否存在棋子
        for(int i = row-1,j = col-1;i >= 0 && j >= 0;i--,j--){
            if(board[i][j] == 'Q') return false;
        }
        // 判断右上方是否存在棋子
        for(int i = row-1,j = col+1;i >= 0 && j < board.length;i--,j++){
            if(board[i][j] == 'Q') return false;
        }
        return true;
    }
}
```

## 二分

[35. 搜索插入位置 - 力扣（LeetCode）](https://leetcode.cn/problems/search-insert-position/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length;
        while(left < right){
            int mid = (left + right) >> 1;
            if(nums[mid] == target) return mid;
            else if(nums[mid] < target) left = mid+1;
            else right = mid;
        }
        // 没有找到该元素，此时left位于大于目标值的最小元素索引
        return left;
    }
}
```

[74. 搜索二维矩阵 - 力扣（LeetCode）](https://leetcode.cn/problems/search-a-2d-matrix/description/?envType=study-plan-v2&envId=top-100-liked)

思路：

将二维矩阵的每一行拼接起来就是一个递增的数组，因此可以在区间 [0..row*col-1] 二分搜索目标值，在比较时将一维下标转换为矩阵的坐标，从而获得矩阵的值，按照二分搜索框架修改搜索区间即可。

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length; 
        int left = 0, right = m*n;
        while(left < right){
            int mid = (left + right) >> 1;
            int row = mid/n, col = mid % n;
            if(matrix[row][col] == target) return true;
            else if(matrix[row][col] < target) left = mid+1;
            else right = mid;
        }
        return false;
    }
}
```

[34. 在排序数组中查找元素的第一个和最后一个位置 - 力扣（LeetCode）](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        return new int[]{left_bound(nums, target), right_bound(nums, target)};
    }

    int left_bound(int[] nums, int target){
        int left = 0, right = nums.length;
        while(left < right){
            int mid = (left + right) >> 1;
            if(nums[mid] == target) right = mid;
            else if(nums[mid] < target) left = mid + 1;
            else right = mid;
        }
        // 判断是否超出边界
        if(left >= nums.length || nums[left] != target) return -1;
        return left;
    }
    int right_bound(int[] nums, int target){
        int left = 0, right = nums.length;
        while(left < right){
            int mid = (left + right) >> 1;
            if(nums[mid] == target) left = mid + 1;
            else if(nums[mid] < target) left = mid + 1;
            else right = mid;
        }
        // 判断是否超出边界
        if(right-1 < 0 || nums[right-1] != target) return -1;
        return right-1;
    }
}
```

[33. 搜索旋转排序数组 - 力扣（LeetCode）](https://leetcode.cn/problems/search-in-rotated-sorted-array/?envType=study-plan-v2&envId=top-100-liked)

思路：

设旋转排序数组的中间边界为p，则边界左边的数组和右边的数组都是升序的，且左边数组的最小值大于右边数组的最大值，即a[p+1]<a[p+2]<...<a[n-1]<a[0]<a[1]<...<a[p]。因此可以使用二分搜索，搜索区间为[l, r]，初始搜索区间为[0, n-1]，设中间位置mid = (l+r)/2，分两种情况：

- a[mid] >= a[0]，则mid位于左边，a[0..mid]是升序的，若target位于区间a[0..mid]，则right = mid-1，否则left = mid+1
- a[mid] <= a[n-1]，则mid位于右边，a[mid..n-1]是升序的，若target位于区间a[mid..right]，则left = mid+1，否则right=mid-1

```java
class Solution {
    public int search(int[] nums, int target) {
        int n = nums.length;
        int left = 0, right = nums.length-1;
        while(left <= right){
            int mid = (left + right) >> 1;
            // 找到了目标值，直接返回
            if(nums[mid] == target){
                return mid;
            }
            if(nums[mid] >= nums[0]){ //nums[0..mid]是有序的
                if(nums[0] <= target && target < nums[mid]){ //目标值在有序区间
                    right = mid - 1;
                }else{
                    left = mid + 1;
                }
            }else if(nums[mid] <= nums[n-1]){ //nums[mid..n-1]是有序的
                if(nums[mid] < target && target <= nums[n-1]){ //目标值在有序区间
                    left = mid + 1;
                }else{
                    right = mid - 1;
                }
            }
        }
        return -1;
    }
}
```

[153. 寻找旋转排序数组中的最小值 - 力扣（LeetCode）](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/?envType=study-plan-v2&envId=top-100-liked)

思路：

类似上题，设左右数组的边界位置为p，则最小值的位于p+1，若nums[mid] < nums[right]，说明mid在右边，right = mid，否则mid在左边，left = mid+1。

```java
class Solution {
    public int findMin(int[] nums) {
        int n = nums.length;
        int left = 0, right = n - 1;
        while(left < right){
            int mid = (left + right) >> 1;
            if(nums[mid] < nums[right]) right = mid;
            else left = mid + 1;
        }
        return nums[left];
    }
}
```

## 栈

[20. 有效的括号 - 力扣（LeetCode）](https://leetcode.cn/problems/valid-parentheses/?envType=study-plan-v2&envId=top-100-liked)

思路：

使用栈保存括号，遍历每一个括号，若为左括号，直接入栈；若为右括号，判断栈顶元素是否为左括号且与当前括号匹配，若匹配，将栈顶括号弹出，否则说明括号串是无效的。

```java
class Solution {
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for(char c:s.toCharArray()){
            if(c=='(' || c=='[' || c=='{') stack.push(c);
            else if(!stack.isEmpty() && match(stack.peek(),c)) stack.pop();
            else return false;
        }
        return stack.isEmpty();
    }
    boolean match(char a,char b){
        return a=='{'&&b=='}'||a=='['&&b==']'||a=='('&&b==')';
    }
}
```

[155. 最小栈 - 力扣（LeetCode）](https://leetcode.cn/problems/min-stack/?envType=study-plan-v2&envId=top-100-liked)

```java
class MinStack {
    // 记录栈中所有元素
    Stack<Integer> stk = new Stack<>();
    // 记录栈中每个元素入栈时的元素最小值
    Stack<Integer> minStk = new Stack<>();
    public MinStack() {
        
    }
    
    public void push(int val) {
        stk.push(val);
        // 维护元素入栈后的栈中最小元素
        if(minStk.isEmpty() || val <= minStk.peek()){
            // 入栈元素就是最小元素
            minStk.push(val);
        }
    }
    
    public void pop() {
        int val = stk.pop();
        // 出栈元素是栈中最小元素
        if(val.equals(minStk.peek())){
            minStk.pop();
        }
    }
    
    public int top() {
        return stk.peek();
    }
    
    public int getMin() {
        return minStk.peek();
    }
}
```

[394. 字符串解码 - 力扣（LeetCode）](https://leetcode.cn/problems/decode-string/?envType=study-plan-v2&envId=top-100-liked)

思路一：

使用一个栈 stk 保存字符串，使用 i 记录当前遍历字符的下标，对于不同的字符，分3种情况：

- 字母或左括号：直接入栈
- 数字：解析当前数字，并入栈
- 右括号：不断弹出栈中的字符串，并逆序拼接成串subStr，直到遇到左括号为止，此时栈顶元素一定为数字，将数字 x 弹出，将串subStr重复 x 次重新入栈

最终将栈中的字符串都逆序拼接起来即可。

```java
class Solution {
    // 当前遍历字符的下标
    int i = 0;
    char[] a;
    public String decodeString(String s) {
        // 保存入栈的字符串
        LinkedList<String> stk = new LinkedList<>();
        a = s.toCharArray();

        while(i < a.length){
            if(Character.isLetter(a[i]) || a[i] == '['){
                // 字母或左括号直接入栈
                stk.addLast(String.valueOf(a[i++]));
            }else if(Character.isDigit(a[i])){
                // 获取一个数字并入栈
                stk.addLast(getDigits());
            }else if(a[i] == ']'){
                // 弹出栈中元素并生成新的字符串，直到遇到左括号为止
                LinkedList<String> sub = new LinkedList<>();
                while(!"[".equals(stk.peekLast())){
                    sub.addFirst(stk.removeLast());
                }
                // 弹出左括号
                stk.removeLast();
                // 将字符串重复num次，添加到栈中
                String subStr = getString(sub);
                Integer num = Integer.valueOf(stk.removeLast());
                stk.addLast(subStr.repeat(num));
                i++;
            }
        }
        return getString(stk);
    }

    String getDigits(){
        StringBuilder digit = new StringBuilder();
        while(Character.isDigit(a[i])){
            digit.append(a[i++]);
        }
        return digit.toString();
    }
    String getString(LinkedList<String> list){
        StringBuilder sb = new StringBuilder();
        for(String str:list){
            sb.append(str);
        }
        return sb.toString();
    }
}
```

思路二：

定义递归函数getString()解析字符串，对于当前遍历到的不同字符，有3种情况：

- 数字，则文法为 String -> Digits[String]String，解析数字num，跳过左括号，递归调用getString()解析字符串str，再跳过右括号，将当前str重复num次，递归调用getString()
- 字符，则文法为 String -> Char String，解析字符，递归调用getString()
- 右括号，则文法为 String -> EOF，返回空串即可

```java
class Solution {
    // 当前遍历字符的下标
    int i = 0;
    char[] a;
    public String decodeString(String s) {
        a = s.toCharArray();
        return getString();
    }

    String getString(){
        if(i == a.length || a[i] == ']'){
            // String -> EOF
            return "";
        } 

        String res = "";
        if(Character.isDigit(a[i])){
            // String -> Digits[String]String
            // 解析Digits
            Integer num = Integer.valueOf(getDigits());
            // 跳过左括号
            i++;
            // 解析String
            String str = getString();
            // 跳过右括号
            i++;
            // 构造字符串
            res = str.repeat(num);
        }else if(Character.isLetter(a[i])){
            // String -> Char String
            // 解析Char
            res = String.valueOf(a[i++]);
        }

        // 递归解析剩余的字符串
        return res.concat(getString());
    }

    String getDigits(){
        StringBuilder digit = new StringBuilder();
        while(Character.isDigit(a[i])){
            digit.append(a[i++]);
        }
        return digit.toString();
    }
}
```

[739. 每日温度 - 力扣（LeetCode）](https://leetcode.cn/problems/daily-temperatures/?envType=study-plan-v2&envId=top-100-liked)

思路：

为了得到第 i 天后面第一个更高的温度出现在几天后，可以逆序遍历每天的温度，使用单调栈保存后面比当天温度高的天的下标，当第i天入栈时，将栈中温度低于 temperatures[i] 的天都弹出，则第i天后面第一个更高的温度出现的天的下标为栈顶元素，若栈为空，则不存在比当天更高的温度。

```java
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] res = new int[n];
        // 单调递增栈
        Stack<Integer> s = new Stack<>();
        // 从后面开始遍历
        for(int i = n-1;i >= 0;i--){
            int x = temperatures[i];
            // 将后面温度低于第i天的出栈
            while(!s.isEmpty() && temperatures[s.peek()] <= x) s.pop();
            // 计算后面一个更高温度出现在几天后
            res[i] = s.isEmpty() ? 0 : s.peek() - i;
            // 将当天的索引入栈
            s.push(i);
        }
        return res;
    }
}
```

[84. 柱状图中最大的矩形 - 力扣（LeetCode）](https://leetcode.cn/problems/largest-rectangle-in-histogram/description/?envType=study-plan-v2&envId=top-100-liked)

思路一：

设矩形的的左右边界为left, right，则矩形的面积即为区间中最低的柱子的高度与区间长度之积，可以枚举左右边界，区间中的最低柱子可以一边枚举一边计算。由于需要枚举两个边界，时间复杂度为O(N^2)。

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int res = 0;
        // 枚举矩形的左右边界
        for(int left = 0;left <= n-1;left++){
            int minHeight = heights[left];
            for(int right = left;right <= n-1;right++){
                minHeight = Math.min(minHeight, heights[right]);
                res = Math.max(res, (right - left + 1) * minHeight);
            }
        }
        return res;
    }
}
```

思路二：

类似思路一，枚举矩形的高度，使用第i根柱子作为矩形的高度heights[i]，则左边界为当前柱子左边第一个高度小于heights[i]的柱子，右边界为当前柱子右边第一个高度小于heights[i]的柱子。时间复杂度为O(n^2)。

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int res = 0;
        // 枚举矩形的高度
        for(int mid = 0;mid <= n-1;mid++){
            int left = mid-1, right = mid+1;
            while(left >= 0 && heights[left] >= heights[mid]) left--;
            while(right < n && heights[right] >= heights[mid]) right++;
            res = Math.max(res, (right - left - 1) * heights[mid]);
        }
        return res;
    }
}
```

思路三：

预先将每个位置 i 左边和右边第一个高度小于heights[i]的主子的位置计算出来，保存在left, right数组中。使用单调栈，元素heights[i]入栈时将大于heights[i]的元素出栈，则left[i]即为栈顶元素，right数组的计算同理。最终枚举矩形的高度，求出每个高度的矩形的最大面积，结果取最大值即可。

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        // left[i]为i左侧第一个高度小于heights[i]的矩形的位置
        int[] left = new int[n];
        // right[i]为i右侧第一个高度小于heights[i]的矩形的位置
        int[] right = new int[n];

        // 单调递增栈
        Deque<Integer> mono_stack = new ArrayDeque<>();
        for(int i = 0;i < n;i++){
            // 将不小于heights[i]的元素出栈
            while(!mono_stack.isEmpty() && heights[mono_stack.peek()] >= heights[i]) {
                mono_stack.pop();
            }
            left[i] = mono_stack.isEmpty() ? -1:mono_stack.peek();
            mono_stack.push(i);
        }
        mono_stack.clear();

        for(int i = n-1;i >= 0;i--){
            // 将不小于heights[i]的元素出栈
            while(!mono_stack.isEmpty() && heights[mono_stack.peek()] >= heights[i]){
                mono_stack.pop();
            }
            right[i] = mono_stack.isEmpty() ? n:mono_stack.peek();
            mono_stack.push(i);
        }

        int res = 0;
        for(int i = 0;i<n;i++){
            res = Math.max(res, (right[i] - left[i] - 1) * heights[i]);
        }
        return res;
    }
}
```

[215. 数组中的第K个最大元素 - 力扣（LeetCode）](https://leetcode.cn/problems/kth-largest-element-in-an-array/?envType=study-plan-v2&envId=top-100-liked)

思路一：

使用快速选择算法，定义左右边界lo, hi，使用nums[lo]作为基准元素，找到数组的基准位置p，使得左边元素都小于nums[p]，右边元素都大于等于nums[p]。要找到第k个元素，若 k<=p，则递归调用选择函数，在左区间搜索，否则在右区间搜索。

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        int n = nums.length;
        return quickSelect(nums, 0, n-1, n-k);
    }
    // 快速选择：从nums[lo..hi]中选择第k大的元素
    int quickSelect(int[] nums, int lo, int hi, int k){
        if(lo == hi) return nums[lo];
        // 选择第一个元素作为基准元素
        int x = nums[lo];
        int j = lo;
        for(int i = lo+1;i<=hi;i++){
            // 如果当前元素小于基准元素，交换到左边区间
            if(nums[i] < x){
                j++;
                swap(nums, i, j);
            }
        }
        // 将基准元素移到左右区间的边界处
        swap(nums, lo, j);
        // 根据元素的顺序到指定的区间选择
        if(k <= j) return quickSelect(nums, lo, j, k);
        else return quickSelect(nums, j+1, hi, k);
    }
    void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

思路二：

使用堆排序，先将数组保存到大顶堆中，再弹出堆顶元素k-1次，此时堆顶元素就是第k大的元素。

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        int n = nums.length;
        MyPriorityQueue pq = new MyPriorityQueue((a,b) -> a-b);
        pq.buildHeap(nums);
        for(int i=0;i<k-1;i++) pq.poll();
        return pq.peek();
    }
}
class MyPriorityQueue{
    private final int N = (int)1e5+5;
    // 保存堆中的元素
    private int[] nums;
    // 堆的大小
    private int size;
    // 比较元素的大小（实现大顶堆或小顶堆）
    Comparator<Integer> comparator;

    public MyPriorityQueue(Comparator<Integer> comparator){
        this.size = 0;
        this.comparator = comparator;
    }

    // 初始建堆（将数组中的元素保存在堆中）
    public void buildHeap(int[] arr){
        this.nums = arr;
        this.size = arr.length;
        for(int i = size/2 - 1;i >= 0;i--){
            heapify(i);
        }
    }

    // 添加一个元素
    public void offer(int x){
        nums[size++] = x;
        swap(nums, 0, size - 1);
        for(int i = size/2-1;i >= 0;i--) heapify(i);
        heapify(0);
    }

    // 删除堆顶元素
    public int poll(){
        int x = nums[0];
        swap(nums, 0, --size);
        heapify(0);
        return x;
    }

    // 返回堆顶元素
    public int peek(){
        return nums[0];
    }

    public int size(){
        return size;
    }

    public boolean isEmpty(){
        return size == 0;
    }
    
    // 从节点i开始将调整堆
    private void heapify(int i){
        int l = i * 2 + 1, r = i * 2 + 2;
        int largest = i;
        // 将最大(大顶堆)或最小(小顶堆)的元素与父节点交换
        if(l < size && comparator.compare(nums[l], nums[largest]) > 0) largest = l;
        if(r < size && comparator.compare(nums[r], nums[largest]) > 0) largest = r;
        if(largest != i){
            swap(nums, i, largest);
            // 递归开始调整下一层节点
            heapify(largest);
        }
    }

    private void swap(int[] nums,int i,int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

[347. 前 K 个高频元素 - 力扣（LeetCode）](https://leetcode.cn/problems/top-k-frequent-elements/?envType=study-plan-v2&envId=top-100-liked)

思路：

使用哈希表保存每个元素出现的次数，再将每个元素和出现的次数保存到大顶堆中，按照出现次数降序排序，堆顶元素就是出现次数最多的。

```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        // 大顶堆
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> b[1] - a[1]);
        // 保存每个数字出现的次数
        Map<Integer, Integer> map = new HashMap<>();
        for(int x:nums){
            map.put(x, map.getOrDefault(x, 0) + 1);
        }
        for(int x:map.keySet()){
            pq.offer(new int[]{x, map.get(x)});
        }

        int[] res = new int[k];
        for(int i = 0;i < k;i++){
            res[i] = pq.poll()[0];
        }
        return res;
    }
}
```

[295. 数据流的中位数 - 力扣（LeetCode）](https://leetcode.cn/problems/find-median-from-data-stream/?envType=study-plan-v2&envId=top-100-liked)

思路：

使用一个大顶堆和一个小顶堆保存数据，大顶堆保存升序数组中间位置左边的元素，小顶堆保存中间位置右边的元素，当添加元素时选择size较小的添加，但是需要先添加到另一个堆中，再将另一个堆中的堆顶元素弹出添加到本堆中；当求中位数时，若两个堆的size相等，分别取堆顶元素求平均值即可，否则取size较大的堆的堆顶元素。

```java
class MedianFinder {
    // 大顶堆：(0,mid)
    PriorityQueue<Integer> left=new PriorityQueue<>((a,b)->b-a);
    // 小顶堆: (mid,n-1)
    PriorityQueue<Integer> right=new PriorityQueue<>();
    public MedianFinder() {

    }
    
    public void addNum(int num) {
        // 将num添加到元素数量较小的堆中
        if(left.size() < right.size()){
            right.offer(num);
            left.offer(right.poll());
        }else{
            left.offer(num);
            right.offer(left.poll());
        }
    }
    
    public double findMedian() {
        if(left.size() == right.size()) return (left.peek() + right.peek())/2.0;
        else if(left.size() > right.size()) return left.peek();
        else return right.peek();
    }
}
```

## 贪心

[121. 买卖股票的最佳时机 - 力扣（LeetCode）](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/?envType=study-plan-v2&envId=top-100-liked)

思路一：

使用minprice记录当前股票的最低价格，遍历每一天的股票价格x，则当前最大利润为 x-minprice，与可以获取的最大利润maxprofix取最大值。

```java
class Solution {
    public int maxProfit(int[] prices) {
        // 当前最小价格
        int minprice = Integer.MAX_VALUE;
        // 当前最大利润
        int maxprofit = 0;
        for(int x:prices){
            maxprofit = Math.max(maxprofit, x - minprice);
            minprice = Math.min(minprice, x);
        }
        return maxprofit;
    }
}
```

思路二：

设$dp[i][k][j]$表示第i天可以获得的最大利润，且最大操作次数为k，j表示第i天是否持有股票，则有
$$
dp[i][k][0]=max(dp[i-1][k][0],dp[i-1][k][1]+prices[i])\\
dp[i][k][1]=max(dp[i-1][k][1],dp[i-1][k-1][0]-prices[i])
$$
base case为$dp[0][..][0]=dp[..][0][0]=0,dp[0][..][1]=dp[..][0][1]=-inf$

本题是k=1的特例，则递推公式为
$$
dp[i][0]=max(dp[i-1][0],dp[i-1][1]+prices[i])\\dp[i][1]=max(dp[i-1][1],-prices[i])
$$
base case为$dp[0][0]=0,dp[0][1]=-inf$。

```java
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        // dp[i][0]表示第i天可以获得的利润（第i天不持有股票）
        // dp[i][1]表示第i天可以获得的利润（第i天持有股票）
        int[][] dp = new int[n+1][2];
        dp[0][0] = 0;
        dp[0][1] = Integer.MIN_VALUE;
        for(int i = 1;i <= n;i++){
            // 第i天不操作或售出
            dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices[i-1]);
            // 第i天不操作或买入
            dp[i][1] = Math.max(dp[i-1][1], -prices[i-1]);
        }
        return dp[n][0];
    }
}
```

[55. 跳跃游戏 - 力扣（LeetCode）](https://leetcode.cn/problems/jump-game/?envType=study-plan-v2&envId=top-100-liked)

思路：

使用farthest记录当前能够跳到的最远距离，遍历每一个位置，更新farthest，如果最远距离不大于当前位置，表示无法到达下一个位置，返回false，若都可以到达，返回true。

```java
class Solution {
    public boolean canJump(int[] nums) {
        int n = nums.length;
        // 当前能够跳到的最远距离
        int farthest = 0;
        for(int i = 0;i < n - 1;i++){
            // 更新farthest
            farthest = Math.max(farthest, i + nums[i]);
            // 最远距离不大于i，无法到达下一个位置
            if(farthest <= i) return false;
        }
        // 判断是否可以达到最后一个位置
        return true;
    }
}
```

[45. 跳跃游戏 II - 力扣（LeetCode）](https://leetcode.cn/problems/jump-game-ii/?envType=study-plan-v2&envId=top-100-liked)

思路：

- 使用farthest记录当前可以到达的最远位置，定义end为跳的前一步可以到达的最远位置，jumps为当前跳的步数。
- 对于位置i，可以到达的区间为[i+1, i + nums[i]]，设我们应该跳到的位置为p，则p+nums[p]应该是范围中最大的，在遍历该区间结束时已经保存到farthest中，此时需要跳一步，更新结束位置end = farthest，跳的步数+1。

```java
class Solution {
    public int jump(int[] nums) {
        int n = nums.length;
        // 当前能够跳得最远的位置
        int farthest = 0;
        // 当前跳的这一步可以到达的最远位置
        int end = 0;
        // 当前跳的步数
        int jumps = 0;
        for(int i = 0;i < n - 1;i++){
            // 更新farthest
            farthest = Math.max(farthest, i + nums[i]);
            // 上一步可以到达的范围为[i+1,end]
            if(i == end){
                end = farthest;
                jumps++;
            }
        }
        return jumps;
    }
}
```

[763. 划分字母区间 - 力扣（LeetCode）](https://leetcode.cn/problems/partition-labels/?envType=study-plan-v2&envId=top-100-liked)

思路一：

首先讲字符串的每个字符出现的次数保存在cnt1中，遍历字符串s，寻找合法的子串，每次将当前遍历的字符的出现次数保存到cnt2中，当cnt2中每个字符出现的次数与cnt1中对应字符出现次数相等，则表明该子串所有的字符只出现在该子串中，添加到结果中。

```java
class Solution {
    public List<Integer> partitionLabels(String s) {
        // 保存字符串s中每个字符出现的次数
        int[] cnt1 = new int[26];
        char[] a = s.toCharArray();
        for(char c:a) {
            cnt1[c-'a']++;
        }

        List<Integer> res = new ArrayList<>();
        int i = 0;
        // 遍历字符串s
        while(i < a.length){
            // 保存当前子串的每个字符出现的次数
            int[] cnt2 = new int[26];
            // 找到子串的结束位置
            int j = i;
            while(j < a.length){
                cnt2[a[j]-'a']++;
                // 找到了一个合法子串
                if(check(cnt1, cnt2)) {
                    j++;
                    break;
                }
                j++;
            }
            res.add(j - i);
            // 更新下一个字符的位置
            i = j;
        }

        return res;
    }
    boolean check(int[] cnt1, int[] cnt2){
        for(int i = 0;i < 26;i++){
            // 子串中每个字符必须不出现在其它子串中
            if(cnt2[i] > 0 && cnt2[i] != cnt1[i]) return false;
        }
        return true;
    }
}
```

思路二：

使用last[26]数组记录每个字符最后一次出现的位置，使用start记录字串的开始位置，end记录子串中每个字符最后一次出现位置的最大值，遍历字符串s，当end==i时，说明子串中所有字符都只出现在本子串中，添加到结果中。

```java
class Solution {
    public List<Integer> partitionLabels(String s) {
        // 记录每个字符最后一次出现的位置
        int[] last = new int[26];
        char[] a = s.toCharArray();
        for(int i = 0;i<a.length;i++) {
            last[a[i]-'a'] = i;
        }

        List<Integer> res = new ArrayList<>();
        int i = 0;
        // 每个子串的开始和结束位置
        int start = 0;
        int end = 0;
        // 遍历字符串s
        while(i < a.length){
            // 更新end
            end = Math.max(end, last[a[i] - 'a']);
            // 到达结束位置，子串的所有字符都是只包含在当前子串中
            if(end == i){
                res.add(end - start + 1);
                start = end + 1;
            }
            i++;
        }

        return res;
    }
}
```

## 动态规划

[70. 爬楼梯 - 力扣（LeetCode）](https://leetcode.cn/problems/climbing-stairs/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int climbStairs(int n) {
        int[] f = new int[n + 1];
        f[0] = 1;
        f[1] = 1;
        for(int i = 2;i <= n;i++){
            f[i] = f[i-1] + f[i-2];
        }
        return f[n];
    }
}
```

[118. 杨辉三角 - 力扣（LeetCode）](https://leetcode.cn/problems/pascals-triangle/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        // 先将第一行加入
        List<List<Integer>> firstRow = new ArrayList<>();
        firstRow.add(1);
        res.add(firstRow);
        for(int i = 2;i <= numRows;i++){
            List<Integer> row = new ArrayList<>();
            row.add(1);
            List<Integer> pre = res.get(res.size() - 1);
            // 讲前一行的每两个元素相加添加到新行中
            for(int j = 0;j<pre.size() - 1;j++){
                row.add(pre.get(j) + pre.get(j+1));
            }
            row.add(1);
            res.add(row);
        }
        return res;
    }
}
```

[198. 打家劫舍 - 力扣（LeetCode）](https://leetcode.cn/problems/house-robber/?envType=study-plan-v2&envId=top-100-liked)

定义f[i]表示前i间房间可以获得的最高金额，对于第i间房，可以选或不选，若选则相邻房屋不能选，否则可以选相邻房屋，递归方程为$f[i]=max(f[i-1],f[i-2]+nums[i-1])$，base case 为$f[0]=0,f[1]=nums[0]$。

```java
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        // f[i]表示前i间房屋可以获得的最高金额
        int[] f = new int[n+1];
        //base case
        f[0] = 0;
        f[1] = nums[0];
        for(int i = 2;i<=n;i++){
            f[i] = Math.max(f[i-1], f[i-2] + nums[i-1]);
        }
        return f[n];
    }
}
```

[279. 完全平方数 - 力扣（LeetCode）](https://leetcode.cn/problems/perfect-squares/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int numSquares(int n) {
        //f[i]表示和为i的完全平方数的最小数量
        int[] f = new int[n+1];
        Arrays.fill(f, Integer.MAX_VALUE);
        f[0] = 0;
        for(int i = 1;i<=n;i++){
            //枚举每一个可能的平方数
            for(int j = 1;j*j<=i;j++){
                f[i] = Math.min(f[i], f[i-j*j] + 1);
            }
        }

        return f[n];
    }
}
```

[322. 零钱兑换 - 力扣（LeetCode）](https://leetcode.cn/problems/coin-change/description/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        // f[i]表示凑成金额i需要最少的硬币数
        int[] f = new int[amount+1];
        int inf = (int)1e4+1;
        Arrays.fill(f,inf);
        // 凑成金额0只需要0个金币
        f[0] = 0;
        for(int i = 1;i <= amount;i++){
            // 枚举可以选的金币
            for(int coin:coins){
                if(coin <= i) f[i] = Math.min(f[i], f[i-coin] + 1);
            }
        }
        return f[amount] == inf ? -1:f[amount];
    }
}
```

[139. 单词拆分 - 力扣（LeetCode）](https://leetcode.cn/problems/word-break/?envType=study-plan-v2&envId=top-100-liked)

思路一：

定义f[i]表示子串s[0..i-1]是否可以被字典中的单词拼接出，可以枚举字典中每一个单词，若单词word为字符串s的后缀，则字符串能否被单词拼接转换为字符串去掉后面的单词后能否被单词拼接，则递推方程为
$$
f[i]=f[i-len_1](word_1=a[i-len,i-1])|f[i-len_2](word_2=a[i-len_2,i-1])|..
$$

```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        char[] a = s.toCharArray();
        int n = a.length;
        // f[i]表示子串s[0..i-1]是否可以被字典中的单词拼接出
        boolean[] f = new boolean[n + 1];
        // base case
        f[0] = true;
        for(int i = 1;i <= n;i++){
            // 枚举每一个单词
            for(String word:wordDict){
                int len = word.length();
                // 判断单词是否为子串s[0..i-1]的后缀
                if(i >= len && word.equals(s.substring(i-len, i))){
                    f[i] |= f[i-len];
                    // 当前子串可以被拼接出，结束枚举
                    if(f[i]) break;
                }
            }
        }
        return f[n];
    }
}
```

[300. 最长递增子序列 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-increasing-subsequence/?envType=study-plan-v2&envId=top-100-liked)

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        // f[i]表示以nums[i]结尾的递增子序列的长度
        int[] f = new int[n];
        for(int i = 0;i<n;i++){
            f[i] = 1;
            // 在前面枚举每一个小于nums[i]的元素
            for(int j = i-1;j>=0;j--){
                if(nums[j] < nums[i]) f[i] = Math.max(f[i], f[j] + 1);
            }
        }

        // 遍历以每个位置元素结尾的递增序列，取长度最大的
        int res = 0;
        for(int i = 0;i < n;i++){
            res = Math.max(res, f[i]);
        }
        return res;
    }
}
```

[152. 乘积最大子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-product-subarray/?envType=study-plan-v2&envId=top-100-liked)

思路：

定义f1[i]表示以nums[i-1]结尾的子数组的最大乘积，f2[i]表示以nums[i-1]结尾的子数组的最小乘积，则对于每个数字x，有三种选择：

- 与前面乘积最大的子数组结合(一般x为正数)
- 与前面乘积最小的子数组结合(一般x为负数)
- 自己单独成为一个子数组(x=0或单独成为一个数时乘积更大)

则递推方程为：
$$
f1[i] = max(f1[i-1] * nums[i-1], f2[i-1] * nums[i-1], nums[i-1])\\
f2[i] = min(f2[i-1] * nums[i-1], f1[i-1] * nums[i-1], nums[i-1])
$$

```java
class Solution {
    public int maxProduct(int[] nums) {
        int n = nums.length;
        // f1[i]：以nums[i-1]结尾的子数组的最大乘积
        int[] f1 = new int[n+1];
        // f2[i]：以nums[i-1]结尾的子数组的最小乘积
        int[] f2 = new int[n+1];
        // base case
        f1[0] = 1;
        f2[0] = 1;

        int res = Integer.MIN_VALUE;
        for(int i = 1;i <= n;i++){
            f1[i] = max(f1[i-1] * nums[i-1], f2[i-1] * nums[i-1], nums[i-1]);
            f2[i] = min(f2[i-1] * nums[i-1], f1[i-1] * nums[i-1], nums[i-1]);
            res = max(res, f1[i], f2[i]);
        }
        return res;
    }
    int max(int a,int b,int c){
        return Math.max(a, Math.max(b, c));
    }
    int min(int a,int b,int c){
        return Math.min(a, Math.min(b, c));
    }
}
```

[416. 分割等和子集 - 力扣（LeetCode）](https://leetcode.cn/problems/partition-equal-subset-sum/?envType=study-plan-v2&envId=top-100-liked)

思路：

首先求出数组和的一般作为目标值target，则题目转换为判断在nums中是否存在和为target的选择方案，即为背包问题。

```java
class Solution {
    public boolean canPartition(int[] nums) {
        // 目标值为数组元素之和的一半
        int target = 0;
        for(int x:nums){
            target += x;
        }
        if(target%2 != 0) return false;
        target /= 2;

        int n = nums.length;
        // 定义f[i][j]: 只选前i个元素，能否使目标值为j
        boolean[][] f = new boolean[n+1][target+1];
        // base case
        for(int i = 0;i<=n;i++) f[i][0] = true;
        for(int i = 1;i<=n;i++){
            for(int j = 1;j<=target;j++){
                if(j >= nums[i-1]) f[i][j] = f[i-1][j] | f[i-1][j-nums[i-1]];
                else f[i][j] = f[i-1][j];
            }
        }
        return f[n][target];
    }
}
```

[32. 最长有效括号 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-valid-parentheses/?envType=study-plan-v2&envId=top-100-liked)

思路：

定义f[i]记录以s[i-1]结尾的最长合法括号子串长度，使用栈stk记录栈中左括号的的下标，遍历字符串s，分两种情况：

- 字符为左括号，左括号不可能是合法括号串，直接入栈，并记录f[i]=0
- 字符为右括号，若栈中存在左括号，将一个左括号的下标 leftIdx 出栈，则f[i] = f[leftIdx-1] + i - leftIdx + 1；若栈中不存在左括号，则表示不能形成有效括号子串，f[i] = 0

```java
class Solution {
    public int longestValidParentheses(String s) {
        char[] a = s.toCharArray();
        int n = a.length;
        // 定义f[i]: 记录以s[i-1]结尾的最长合法括号子串长度 
        int[] f = new int[n+1];
        Stack<Integer> stk = new Stack<>();

        int res = 0;
        for(int i = 1;i <= n;i++){
            char c = a[i-1];
            if(c == '(') {
                // 遇到左括号，记录索引
                stk.push(i);
                // 左括号不可能是合法括号子串的结尾
                f[i+1] = 0;
            }else{
                // 遇到右括号
                if(!stk.isEmpty()){
                    // 配对的左括号索引
                    int leftIdx = stk.pop();
                    // 计算以该括号结尾的最长有效括号子串的长度
                    f[i] = f[leftIdx - 1] + i - leftIdx + 1;
                    res = Math.max(res, f[i]);
                }else{
                    // 不存在配对的左括号
                    f[i] = 0;
                }
            }
        }

        return res;
    }   
}
```

